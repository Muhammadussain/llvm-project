//===- LowerContractionToRISCVIMEVMADOTPattern.cpp - Lower vector.contract
// to riscv_ime.vmadot --------------------------------------------------===//
//
// Unrolls vector.contract (i8 x i8 -> i32) into 4x4x8 riscv_ime.vmadot ops.
//
// Atom: M=4, N=4, K=8.
//   acc  = 4x4 (16 i32, M2)
//   vs1  = 4x8 (32 i8, M1)
//   vs2  = 4x8 (32 i8, M1)
//
// No padding needed — hardware naturally handles 4x4x8 tile.
//
// Supports BOTH RHS layouts:
//   Case A (canonical): LHS (m,k), RHS (n,k) [transposed], ACC (m,n)
//   Case B (IREE):      LHS (m,k), RHS (k,n) [non-transposed], ACC (m,n)
//
// For Case B, a vector.transpose is inserted on the RHS tile.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Dialect/RISCVIME/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::riscv_ime;

namespace {

// Hardware atom dimensions (VLEN=256, SEW=8)
constexpr int64_t kM0 = 4;
constexpr int64_t kN0 = 4;
constexpr int64_t kK0 = 8;

//===----------------------------------------------------------------------===//
// Helper: peel arith.extsi to reach i8 producer.
//===----------------------------------------------------------------------===//
static Value peelExtSI(Value v) {
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
    auto inTy = dyn_cast<VectorType>(ext.getIn().getType());
    if (inTy && inTy.getElementType().isInteger(8))
      return ext.getIn();
  }
  return v;
}

//===----------------------------------------------------------------------===//
// Helper: extract 2D tile.
//===----------------------------------------------------------------------===//
static Value extractTile(PatternRewriter &rewriter, Location loc, Value src,
                         int64_t r, int64_t c, int64_t rs, int64_t cs) {
  return vector::ExtractStridedSliceOp::create(
      rewriter, loc, src, ArrayRef<int64_t>{r, c},
      ArrayRef<int64_t>{rs, cs}, ArrayRef<int64_t>{1, 1});
}

//===----------------------------------------------------------------------===//
// Helper: insert 2D tile.
//===----------------------------------------------------------------------===//
static Value insertTile(PatternRewriter &rewriter, Location loc, Value tile,
                        Value dst, int64_t r, int64_t c) {
  return vector::InsertStridedSliceOp::create(
      rewriter, loc, tile, dst, ArrayRef<int64_t>{r, c},
      ArrayRef<int64_t>{1, 1});
}

//===----------------------------------------------------------------------===//
// Pattern
//===----------------------------------------------------------------------===//
struct LowerContractionToRISCVIMEVMADOTPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // --- 1. kind = ADD ---
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(op, "non-ADD kind");

    // --- 2. iterators [parallel, parallel, reduction] ---
    auto iters = op.getIteratorTypes();
    if (iters.size() != 3)
      return rewriter.notifyMatchFailure(op, "iterator count != 3");

    auto isPar = [&](unsigned i) {
      return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
             vector::IteratorType::parallel;
    };
    auto isRed = [&](unsigned i) {
      return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
             vector::IteratorType::reduction;
    };
    if (!isPar(0) || !isPar(1) || !isRed(2))
      return rewriter.notifyMatchFailure(op, "iterator types mismatch");

    // --- 3. indexing maps: support BOTH RHS layouts ---
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 || maps[0].getNumResults() != 2 ||
        maps[1].getNumResults() != 2 || maps[2].getNumResults() != 2)
      return rewriter.notifyMatchFailure(op, "non-2D map");

    if (maps[0].getResult(0) != maps[2].getResult(0))
      return rewriter.notifyMatchFailure(op, "LHS/ACC m mismatch");

    bool rhsNeedsTranspose = false;
    if (maps[1].getResult(0) == maps[2].getResult(1) &&
        maps[1].getResult(1) == maps[0].getResult(1)) {
      // RHS map = (n, k): canonical
      rhsNeedsTranspose = false;
    } else if (maps[1].getResult(0) == maps[0].getResult(1) &&
               maps[1].getResult(1) == maps[2].getResult(1)) {
      // RHS map = (k, n): IREE default — needs transpose
      rhsNeedsTranspose = true;
    } else {
      return rewriter.notifyMatchFailure(op, "unsupported RHS layout");
    }

    // --- 4. types: i8 x i8 -> i32 (allow extsi producer) ---
    Value lhs = peelExtSI(op.getLhs());
    Value rhs = peelExtSI(op.getRhs());
    Value acc = op.getAcc();

    auto lhsTy = dyn_cast<VectorType>(lhs.getType());
    auto rhsTy = dyn_cast<VectorType>(rhs.getType());
    auto accTy = dyn_cast<VectorType>(acc.getType());
    if (!lhsTy || !rhsTy || !accTy)
      return rewriter.notifyMatchFailure(op, "non-vector operand");

    if (!lhsTy.getElementType().isInteger(8) ||
        !rhsTy.getElementType().isInteger(8) ||
        !accTy.getElementType().isInteger(32))
      return rewriter.notifyMatchFailure(op, "type mismatch");

    // --- 5. rank 2 ---
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || accTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "non-2D operand");

    // --- 6. not scalable ---
    if (lhsTy.isScalable() || rhsTy.isScalable() || accTy.isScalable())
      return rewriter.notifyMatchFailure(op, "scalable not supported");

    // --- 7. shape divisible by atom (layout-aware) ---
    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N, Kr;
    if (rhsNeedsTranspose) {
      Kr = rhsTy.getDimSize(0);
      N = rhsTy.getDimSize(1);
    } else {
      N = rhsTy.getDimSize(0);
      Kr = rhsTy.getDimSize(1);
    }

    if (K != Kr)
      return rewriter.notifyMatchFailure(op, "K mismatch");
    if (M % kM0 != 0 || N % kN0 != 0 || K % kK0 != 0)
      return rewriter.notifyMatchFailure(op, "shape not divisible by 4x4x8");
    if (accTy.getDimSize(0) != M || accTy.getDimSize(1) != N)
      return rewriter.notifyMatchFailure(op, "ACC shape mismatch");

    //================= REWRITE =================//
    Location loc = op.getLoc();
    Type i8Ty = rewriter.getI8Type();
    Type i32Ty = rewriter.getI32Type();

    // 1D register types (VLEN=256, SEW=8)
    //   acc: 4x4 i32 → 16 x i32
    //   LHS: 4x8 i8  → 32 x i8
    //   RHS: 4x8 i8  → 32 x i8
    auto accTileTy = VectorType::get({kM0, kN0}, i32Ty);  // 16
    auto a1dTy = VectorType::get({kM0 * kK0}, i8Ty);      // 32
    auto b1dTy = VectorType::get({kN0 * kK0}, i8Ty);      // 32
    auto c1dTy = VectorType::get({kM0 * kN0}, i32Ty);     // 16

    Value result = acc;

    for (int64_t m = 0; m < M; m += kM0) {
      for (int64_t n = 0; n < N; n += kN0) {
        // Extract 4x4 acc tile.
        Value accTile = extractTile(rewriter, loc, result, m, n, kM0, kN0);

        // K loop: chain vmadot (acc carries over).
        for (int64_t k = 0; k < K; k += kK0) {
          // LHS tile: 4x8
          Value aTile = extractTile(rewriter, loc, lhs, m, k, kM0, kK0);

          // RHS tile: get 4x8 (no padding needed)
          Value bTile;
          if (rhsNeedsTranspose) {
            // RHS is (K, N): extract 8x4 then transpose to 4x8
            Value bRaw = extractTile(rewriter, loc, rhs, k, n, kK0, kN0);
            bTile = vector::TransposeOp::create(
                rewriter, loc, bRaw, ArrayRef<int64_t>{1, 0});
          } else {
            // RHS is (N, K): extract 4x8 directly
            bTile = extractTile(rewriter, loc, rhs, n, k, kN0, kK0);
          }

          // Shape cast 2D -> 1D
          Value a1d = vector::ShapeCastOp::create(rewriter, loc, a1dTy, aTile);
          Value b1d = vector::ShapeCastOp::create(rewriter, loc, b1dTy, bTile);
          Value c1d = vector::ShapeCastOp::create(rewriter, loc, c1dTy, accTile);

          // vmadot (chained across K)
          Value newC1d = VmadotOp::create(rewriter, loc, c1dTy, c1d, a1d, b1d);

          // Shape cast 1D -> 2D
          accTile = vector::ShapeCastOp::create(rewriter, loc, accTileTy, newC1d);
        }

        // Insert 4x4 acc tile back into result.
        result = insertTile(rewriter, loc, accTile, result, m, n);
      }
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//
struct LowerContractionToRISCVIMEPass
    : public PassWrapper<LowerContractionToRISCVIMEPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerContractionToRISCVIMEPass)

  StringRef getArgument() const final {
    return "lower-contraction-to-riscv-ime";
  }
  StringRef getDescription() const final {
    return "Lower vector.contract (i8 x i8 -> i32) to riscv_ime.vmadot";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<riscv_ime::RISCVIMEDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LowerContractionToRISCVIMEVMADOTPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory + registration
//===----------------------------------------------------------------------===//
std::unique_ptr<::mlir::Pass>
mlir::riscv_ime::createLowerContractionToRISCVIMEPass() {
  return std::make_unique<LowerContractionToRISCVIMEPass>();
}

void mlir::riscv_ime::registerLowerContractionToRISCVIMEPass() {
  PassRegistration<LowerContractionToRISCVIMEPass>();
}
