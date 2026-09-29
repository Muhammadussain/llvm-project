// //===- LowerContractionToRISCVIMEVMADOTPattern.cpp - Lower vector.contract
// // to riscv_ime.vmadot --------------------------------------------------===//
// //
// // Stage 1: exact atom match (4x8 x 4x8 -> 4x4)
// // =============================================================
// //
// // Matches: vector.contract
// //   - kind = ADD
// //   - iterators = [parallel, parallel, reduction]
// //   - maps: LHS (m,k), RHS (n,k) TRANSPOSED, ACC (m,n)
// //   - types: i8 x i8 -> i32
// //   - exact shape: LHS 4x8, RHS 4x8, ACC 4x4
// //
// // Rewrites to: riscv_ime.vmadot
// //   - shape_cast 2D -> 1D
// //   - create VmadotOp
// //   - shape_cast 1D -> 2D back
// //===----------------------------------------------------------------------===//

// #include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
// #include "mlir/Dialect/RISCVIME/Transforms/Passes.h"

// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// using namespace mlir;
// using namespace mlir::riscv_ime;

// namespace {

// //===----------------------------------------------------------------------===//
// // Helper: peel arith.extsi to reach the i8 producer.
// //
// // TCP PDF says "ensure arith.extsi is present" — but that's misleading.
// // vmadot natively widens i8 -> i32. So if the contract operands are already
// // extended to i32, we walk back to the original i8 value.
// //===----------------------------------------------------------------------===//
// static Value peelExtSI(Value v) {
//   if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
//     auto inTy = dyn_cast<VectorType>(ext.getIn().getType());
//     if (inTy && inTy.getElementType().isInteger(8))
//       return ext.getIn();
//   }
//   return v;
// }

// //===----------------------------------------------------------------------===//
// // Pattern
// //===----------------------------------------------------------------------===//
// struct LowerContractionToRISCVIMEVMADOTPattern
//     : public OpRewritePattern<vector::ContractionOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(vector::ContractionOp op,
//                                 PatternRewriter &rewriter) const override {
//     // --- 1. kind = ADD ---
//     if (op.getKind() != vector::CombiningKind::ADD)
//       return rewriter.notifyMatchFailure(op, "non-ADD kind");

//     // --- 2. iterators: [parallel, parallel, reduction] ---
//     auto iters = op.getIteratorTypes();
//     if (iters.size() != 3)
//       return rewriter.notifyMatchFailure(op, "iterator count != 3");

//     auto isPar = [&](unsigned i) {
//       return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
//              vector::IteratorType::parallel;
//     };
//     auto isRed = [&](unsigned i) {
//       return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
//              vector::IteratorType::reduction;
//     };
//     if (!isPar(0) || !isPar(1) || !isRed(2))
//       return rewriter.notifyMatchFailure(op, "iterator types mismatch");

//     // --- 3. indexing maps: LHS (m,k), RHS (n,k), ACC (m,n) ---
//     auto maps = op.getIndexingMapsArray();
//     if (maps.size() != 3)
//       return rewriter.notifyMatchFailure(op, "map count != 3");

//     if (maps[0].getNumResults() != 2 || maps[1].getNumResults() != 2 ||
//         maps[2].getNumResults() != 2)
//       return rewriter.notifyMatchFailure(op, "non-2D map");

//     // LHS (d0, d2) : m, k
//     // RHS (d1, d2) : n, k   <- TRANSPOSED (IME needs (n,k), not (k,n))
//     // ACC (d0, d1) : m, n
//     if (maps[0].getResult(0) != maps[2].getResult(0))  // m
//       return rewriter.notifyMatchFailure(op, "LHS/ACC m mismatch");
//     if (maps[1].getResult(0) != maps[2].getResult(1))  // n
//       return rewriter.notifyMatchFailure(op, "RHS/ACC n mismatch");
//     if (maps[0].getResult(1) != maps[1].getResult(1))  // k
//       return rewriter.notifyMatchFailure(op, "LHS/RHS k mismatch");

//     // --- 4. types: i8 x i8 -> i32 (allow extsi producer) ---
//     Value lhs = peelExtSI(op.getLhs());
//     Value rhs = peelExtSI(op.getRhs());
//     Value acc = op.getAcc();

//     auto lhsTy = dyn_cast<VectorType>(lhs.getType());
//     auto rhsTy = dyn_cast<VectorType>(rhs.getType());
//     auto accTy = dyn_cast<VectorType>(acc.getType());
//     if (!lhsTy || !rhsTy || !accTy)
//       return rewriter.notifyMatchFailure(op, "non-vector operand");

//     if (!lhsTy.getElementType().isInteger(8) ||
//         !rhsTy.getElementType().isInteger(8) ||
//         !accTy.getElementType().isInteger(32))
//       return rewriter.notifyMatchFailure(op, "type mismatch (need i8/i8->i32)");

//     // --- 5. rank 2 ---
//     if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || accTy.getRank() != 2)
//       return rewriter.notifyMatchFailure(op, "non-2D operand");

//     // --- 6. exact atom shape: 4x8, 4x8, 4x4 ---
//     if (lhsTy.getShape() != ArrayRef<int64_t>{4, 8})
//       return rewriter.notifyMatchFailure(op, "LHS shape != 4x8");
//     if (rhsTy.getShape() != ArrayRef<int64_t>{4, 8})
//       return rewriter.notifyMatchFailure(op, "RHS shape != 4x8");
//     if (accTy.getShape() != ArrayRef<int64_t>{4, 4})
//       return rewriter.notifyMatchFailure(op, "ACC shape != 4x4");

//     // --- 7. not scalable ---
//     if (lhsTy.isScalable() || rhsTy.isScalable() || accTy.isScalable())
//       return rewriter.notifyMatchFailure(op, "scalable vector not supported");

//     //================= REWRITE =================//
//     Location loc = op.getLoc();

//     // Target 1D types (VLEN=256, SEW=8)
//     auto lhs1DTy = VectorType::get({32}, rewriter.getI8Type());
//     auto rhs1DTy = VectorType::get({32}, rewriter.getI8Type());
//     auto acc1DTy = VectorType::get({16}, rewriter.getI32Type());

// Value lhs1D = vector::ShapeCastOp::create(rewriter, loc, lhs1DTy, lhs);
// Value rhs1D = vector::ShapeCastOp::create(rewriter, loc, rhs1DTy, rhs);
// Value acc1D = vector::ShapeCastOp::create(rewriter, loc, acc1DTy, acc);
// Value result1D = VmadotOp::create(rewriter, loc, acc1DTy, acc1D, lhs1D, rhs1D);
// Value result2D = vector::ShapeCastOp::create(rewriter, loc, accTy, result1D);

//     // Replace original contract with the 2D result
//     rewriter.replaceOp(op, result2D);
//     return success();
//   }
// };

// //===----------------------------------------------------------------------===//
// // Pass
// //===----------------------------------------------------------------------===//
// struct LowerContractionToRISCVIMEPass
//     : public PassWrapper<LowerContractionToRISCVIMEPass,
//                          OperationPass<func::FuncOp>> {
//   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerContractionToRISCVIMEPass)

//   StringRef getArgument() const final {
//     return "lower-contraction-to-riscv-ime";
//   }
//   StringRef getDescription() const final {
//     return "Lower vector.contract (i8 x i8 -> i32) to riscv_ime.vmadot";
//   }

//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<vector::VectorDialect>();
//     registry.insert<arith::ArithDialect>();
//     registry.insert<riscv_ime::RISCVIMEDialect>();
//   }

//   void runOnOperation() override {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<LowerContractionToRISCVIMEVMADOTPattern>(&getContext());

//         if (failed(applyPatternsGreedily(getOperation(),
//                                      std::move(patterns))))
//       signalPassFailure();
//   }
// };

// } // namespace

// //===----------------------------------------------------------------------===//
// // Registration
// //===----------------------------------------------------------------------===//
// void mlir::riscv_ime::registerLowerContractionToRISCVIMEPass() {
//   PassRegistration<LowerContractionToRISCVIMEPass>();
// }



//===- LowerContractionToRISCVIMEVMADOTPattern.cpp - Lower vector.contract
// to riscv_ime.vmadot -------------------------------------------------===//
//
// Stage 2: unroll larger shapes into 4x8x4 atoms.
// ======================================================
//
// Matches: vector.contract
//   - kind = ADD
//   - iterators = [parallel, parallel, reduction]
//   - maps: LHS (m,k), RHS (n,k) TRANSPOSED, ACC (m,n)
//   - types: i8 x i8 -> i32
//   - M % 4 == 0, N % 4 == 0, K % 8 == 0
//
// Rewrites to: M/4 x N/4 x K/8 riscv_ime.vmadot ops
//   - outer M/N loops: extract 4x4 acc tile
//   - inner K loop: chain vmadot (acc carries over)
//   - insert acc tile back
//===----------------------------------------------------------------------===//

// #include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
// #include "mlir/Dialect/RISCVIME/Transforms/Passes.h"

// #include "mlir/Dialect/Arith/IR/Arith.h"
// #include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Vector/IR/VectorOps.h"
// #include "mlir/IR/PatternMatch.h"
// #include "mlir/Pass/Pass.h"
// #include "mlir/Transforms/GreedyPatternRewriteDriver.h"

// using namespace mlir;
// using namespace mlir::riscv_ime;

// namespace {

// // Atom dimensions (hardware, VLEN=256, SEW=8, LMUL=1)
// constexpr int64_t kAtomM = 4;
// constexpr int64_t kAtomN = 4;
// constexpr int64_t kAtomK = 8;

// //===----------------------------------------------------------------------===//
// // Helper: peel arith.extsi to reach the i8 producer.
// //===----------------------------------------------------------------------===//
// static Value peelExtSI(Value v) {
//   if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
//     auto inTy = dyn_cast<VectorType>(ext.getIn().getType());
//     if (inTy && inTy.getElementType().isInteger(8))
//       return ext.getIn();
//   }
//   return v;
// }

// //===----------------------------------------------------------------------===//
// // Pattern
// //===----------------------------------------------------------------------===//
// struct LowerContractionToRISCVIMEVMADOTPattern
//     : public OpRewritePattern<vector::ContractionOp> {
//   using OpRewritePattern::OpRewritePattern;

//   LogicalResult matchAndRewrite(vector::ContractionOp op,
//                                 PatternRewriter &rewriter) const override {
//     // --- 1. kind = ADD ---
//     if (op.getKind() != vector::CombiningKind::ADD)
//       return rewriter.notifyMatchFailure(op, "non-ADD kind");

//     // --- 2. iterators: [parallel, parallel, reduction] ---
//     auto iters = op.getIteratorTypes();
//     if (iters.size() != 3)
//       return rewriter.notifyMatchFailure(op, "iterator count != 3");

//     auto isPar = [&](unsigned i) {
//       return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
//              vector::IteratorType::parallel;
//     };
//     auto isRed = [&](unsigned i) {
//       return cast<vector::IteratorTypeAttr>(iters[i]).getValue() ==
//              vector::IteratorType::reduction;
//     };
//     if (!isPar(0) || !isPar(1) || !isRed(2))
//       return rewriter.notifyMatchFailure(op, "iterator types mismatch");

//     // --- 3. indexing maps ---
//     auto maps = op.getIndexingMapsArray();
//     if (maps.size() != 3)
//       return rewriter.notifyMatchFailure(op, "map count != 3");

//     if (maps[0].getNumResults() != 2 || maps[1].getNumResults() != 2 ||
//         maps[2].getNumResults() != 2)
//       return rewriter.notifyMatchFailure(op, "non-2D map");

//     // LHS (d0, d2), RHS (d1, d2), ACC (d0, d1)
//     if (maps[0].getResult(0) != maps[2].getResult(0)) return failure();
//     if (maps[1].getResult(0) != maps[2].getResult(1)) return failure();
//     if (maps[0].getResult(1) != maps[1].getResult(1)) return failure();

//     // --- 4. types: i8 x i8 -> i32 ---
//     Value lhs = peelExtSI(op.getLhs());
//     Value rhs = peelExtSI(op.getRhs());
//     Value acc = op.getAcc();

//     auto lhsTy = dyn_cast<VectorType>(lhs.getType());
//     auto rhsTy = dyn_cast<VectorType>(rhs.getType());
//     auto accTy = dyn_cast<VectorType>(acc.getType());
//     if (!lhsTy || !rhsTy || !accTy)
//       return rewriter.notifyMatchFailure(op, "non-vector operand");

//     if (!lhsTy.getElementType().isInteger(8) ||
//         !rhsTy.getElementType().isInteger(8) ||
//         !accTy.getElementType().isInteger(32))
//       return rewriter.notifyMatchFailure(op, "type mismatch");

//     // --- 5. rank 2 ---
//     if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || accTy.getRank() != 2)
//       return rewriter.notifyMatchFailure(op, "non-2D operand");

//     // --- 6. not scalable ---
//     if (lhsTy.isScalable() || rhsTy.isScalable() || accTy.isScalable())
//       return rewriter.notifyMatchFailure(op, "scalable not supported");

//     // --- 7. shape divisible by atom ---
//     int64_t M = lhsTy.getDimSize(0);
//     int64_t K = lhsTy.getDimSize(1);
//     int64_t N = rhsTy.getDimSize(0);
//     int64_t Kr = rhsTy.getDimSize(1);

//     if (K != Kr)
//       return rewriter.notifyMatchFailure(op, "K mismatch");

//     if (M % kAtomM != 0 || N % kAtomN != 0 || K % kAtomK != 0)
//       return rewriter.notifyMatchFailure(op, "shape not divisible by 4x4x8");

//     if (accTy.getDimSize(0) != M || accTy.getDimSize(1) != N)
//       return rewriter.notifyMatchFailure(op, "ACC shape mismatch");

//     //================= REWRITE =================//
//     Location loc = op.getLoc();
//     Type i32Ty = rewriter.getI32Type();

//     Value result = acc;  // accumulate into result

//     for (int64_t m = 0; m < M; m += kAtomM) {
//       for (int64_t n = 0; n < N; n += kAtomN) {
//         // Extract 4x4 acc tile
//         auto accTileTy = VectorType::get({kAtomM, kAtomN}, i32Ty);
//         Value accTile = rewriter.create<vector::ExtractStridedSliceOp>(
//             loc, result, ArrayRef<int64_t>{m, n},
//             ArrayRef<int64_t>{kAtomM, kAtomN}, ArrayRef<int64_t>{1, 1});

//         // K loop: chain vmadot
//         for (int64_t k = 0; k < K; k += kAtomK) {
//           // Extract 4x8 LHS tile
//           Value aTile = rewriter.create<vector::ExtractStridedSliceOp>(
//               loc, lhs, ArrayRef<int64_t>{m, k},
//               ArrayRef<int64_t>{kAtomM, kAtomK}, ArrayRef<int64_t>{1, 1});

//           // Extract 4x8 RHS tile
//           Value bTile = rewriter.create<vector::ExtractStridedSliceOp>(
//               loc, rhs, ArrayRef<int64_t>{n, k},
//               ArrayRef<int64_t>{kAtomN, kAtomK}, ArrayRef<int64_t>{1, 1});

//           // Shape cast 2D -> 1D
//           auto a1dTy = VectorType::get({kAtomM * kAtomK}, rewriter.getI8Type());
//           auto b1dTy = VectorType::get({kAtomN * kAtomK}, rewriter.getI8Type());
//           auto c1dTy = VectorType::get({kAtomM * kAtomN}, i32Ty);

//           Value a1d = vector::ShapeCastOp::create(rewriter, loc, a1dTy, aTile);
//           Value b1d = vector::ShapeCastOp::create(rewriter, loc, b1dTy, bTile);
//           Value c1d = vector::ShapeCastOp::create(rewriter, loc, c1dTy, accTile);

//           // Create VmadotOp (chained across K)
//           Value newC1d =
//               VmadotOp::create(rewriter, loc, c1dTy, c1d, a1d, b1d);

//           // 1D -> 2D
//           accTile = vector::ShapeCastOp::create(rewriter, loc, accTileTy, newC1d);
//         }

//         // Insert 4x4 acc tile back into result
//         result = rewriter.create<vector::InsertStridedSliceOp>(
//             loc, accTile, result, ArrayRef<int64_t>{m, n},
//             ArrayRef<int64_t>{1, 1});
//       }
//     }

//     rewriter.replaceOp(op, result);
//     return success();
//   }

// private:
//   // Helper to get i8 type from the rewriter.
//   static Type i8Ty(PatternRewriter &rewriter) {
//     return rewriter.getI8Type();
//   }
// };

// //===----------------------------------------------------------------------===//
// // Pass
// //===----------------------------------------------------------------------===//
// struct LowerContractionToRISCVIMEPass
//     : public PassWrapper<LowerContractionToRISCVIMEPass,
//                          OperationPass<func::FuncOp>> {
//   MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LowerContractionToRISCVIMEPass)

//   StringRef getArgument() const final {
//     return "lower-contraction-to-riscv-ime";
//   }
//   StringRef getDescription() const final {
//     return "Lower vector.contract (i8 x i8 -> i32) to riscv_ime.vmadot";
//   }

//   void getDependentDialects(DialectRegistry &registry) const override {
//     registry.insert<vector::VectorDialect>();
//     registry.insert<arith::ArithDialect>();
//     registry.insert<riscv_ime::RISCVIMEDialect>();
//   }

//   void runOnOperation() override {
//     RewritePatternSet patterns(&getContext());
//     patterns.add<LowerContractionToRISCVIMEVMADOTPattern>(&getContext());

//     if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
//       signalPassFailure();
//   }
// };

// } // namespace

// //===----------------------------------------------------------------------===//
// // Registration
// //===----------------------------------------------------------------------===//
// void mlir::riscv_ime::registerLowerContractionToRISCVIMEPass() {
//   PassRegistration<LowerContractionToRISCVIMEPass>();
// }
//===- LowerContractionToRISCVIMEVMADOTPattern.cpp - Lower vector.contract
// to riscv_ime.vmadot --------------------------------------------------===//
//
// Unrolls vector.contract (i8 x i8 -> i32) into 4x4x8 riscv_ime.vmadot ops.
//
// Matches:
//   - kind = ADD
//   - iterators = [parallel, parallel, reduction]
//   - maps: LHS (m,k), RHS (n,k) TRANSPOSED, ACC (m,n)
//   - types: i8 x i8 -> i32
//   - M % 4 == 0, N % 4 == 0, K % 8 == 0
//
// Rewrites to: (M/4) x (N/4) x (K/8) riscv_ime.vmadot ops
//   - outer M/N loops: extract 4x4 acc tile
//   - inner K loop: chain vmadot (acc carries over)
//   - insert acc tile back
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Dialect/RISCVIME/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::riscv_ime;

namespace {

//===----------------------------------------------------------------------===//
// Hardware atom dimensions (VLEN=256, SEW=8, LMUL=1)
//===----------------------------------------------------------------------===//
constexpr int64_t kM0 = 4;  // M tile size
constexpr int64_t kN0 = 4;  // N tile size
constexpr int64_t kK0 = 8;  // K tile size

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Peel arith.extsi to reach the i8 producer.
/// vmadot natively widens i8 -> i32, so we walk back to the original i8.
static Value peelExtSI(Value v) {
  if (auto ext = v.getDefiningOp<arith::ExtSIOp>()) {
    auto inTy = dyn_cast<VectorType>(ext.getIn().getType());
    if (inTy && inTy.getElementType().isInteger(8))
      return ext.getIn();
  }
  return v;
}

/// Extract a 2D tile from `src` at position (r, c) with size (rs, cs).
static Value extractTile(PatternRewriter &rewriter, Location loc, Value src,
                         int64_t r, int64_t c, int64_t rs, int64_t cs) {
  return vector::ExtractStridedSliceOp::create(
      rewriter, loc, src, ArrayRef<int64_t>{r, c},
      ArrayRef<int64_t>{rs, cs}, ArrayRef<int64_t>{1, 1});
}

/// Insert `tile` into `dst` at position (r, c).
static Value insertTile(PatternRewriter &rewriter, Location loc, Value tile,
                        Value dst, int64_t r, int64_t c) {
  return vector::InsertStridedSliceOp::create(
      rewriter, loc, tile, dst, ArrayRef<int64_t>{r, c},
      ArrayRef<int64_t>{1, 1});
}

//===----------------------------------------------------------------------===//
// Pattern: vector.contract (i8 x i8 -> i32) -> riscv_ime.vmadot
//===----------------------------------------------------------------------===//
struct LowerContractionToRISCVIMEVMADOTPattern
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    // --- 1. kind = ADD ---
    if (op.getKind() != vector::CombiningKind::ADD)
      return rewriter.notifyMatchFailure(op, "non-ADD kind");

    // --- 2. iterators: [parallel, parallel, reduction] ---
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

    // --- 3. indexing maps: LHS (m,k), RHS (n,k), ACC (m,n) ---
    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 || maps[0].getNumResults() != 2 ||
        maps[1].getNumResults() != 2 || maps[2].getNumResults() != 2)
      return rewriter.notifyMatchFailure(op, "non-2D map");

    if (maps[0].getResult(0) != maps[2].getResult(0))  // m
      return rewriter.notifyMatchFailure(op, "LHS/ACC m mismatch");
    if (maps[1].getResult(0) != maps[2].getResult(1))  // n
      return rewriter.notifyMatchFailure(op, "RHS/ACC n mismatch");
    if (maps[0].getResult(1) != maps[1].getResult(1))  // k
      return rewriter.notifyMatchFailure(op, "LHS/RHS k mismatch");

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
      return rewriter.notifyMatchFailure(op, "type mismatch (need i8/i8->i32)");

    // --- 5. rank 2 ---
    if (lhsTy.getRank() != 2 || rhsTy.getRank() != 2 || accTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "non-2D operand");

    // --- 6. not scalable ---
    if (lhsTy.isScalable() || rhsTy.isScalable() || accTy.isScalable())
      return rewriter.notifyMatchFailure(op, "scalable not supported");

    // --- 7. shape divisible by atom ---
    int64_t M = lhsTy.getDimSize(0);
    int64_t K = lhsTy.getDimSize(1);
    int64_t N = rhsTy.getDimSize(0);
    int64_t Kr = rhsTy.getDimSize(1);

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

    auto accTileTy = VectorType::get({kM0, kN0}, i32Ty);
    auto a1dTy = VectorType::get({kM0 * kK0}, i8Ty);
    auto b1dTy = VectorType::get({kN0 * kK0}, i8Ty);
    auto c1dTy = VectorType::get({kM0 * kN0}, i32Ty);

    Value result = acc;

    for (int64_t m = 0; m < M; m += kM0) {
      for (int64_t n = 0; n < N; n += kN0) {
        // Extract 4x4 acc tile.
        Value accTile = extractTile(rewriter, loc, result, m, n, kM0, kN0);

        // K loop: chain vmadot (acc carries over).
        for (int64_t k = 0; k < K; k += kK0) {
          Value aTile = extractTile(rewriter, loc, lhs, m, k, kM0, kK0);
          Value bTile = extractTile(rewriter, loc, rhs, n, k, kN0, kK0);

          // Shape cast 2D -> 1D.
          Value a1d = vector::ShapeCastOp::create(rewriter, loc, a1dTy, aTile);
          Value b1d = vector::ShapeCastOp::create(rewriter, loc, b1dTy, bTile);
          Value c1d = vector::ShapeCastOp::create(rewriter, loc, c1dTy, accTile);

          // vmadot (chained across K).
          Value newC1d = VmadotOp::create(rewriter, loc, c1dTy, c1d, a1d, b1d);

          // Shape cast 1D -> 2D.
          accTile =
              vector::ShapeCastOp::create(rewriter, loc, accTileTy, newC1d);
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
// Pass: lower-contraction-to-riscv-ime
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
// Registration
//===----------------------------------------------------------------------===//
void mlir::riscv_ime::registerLowerContractionToRISCVIMEPass() {
  PassRegistration<LowerContractionToRISCVIMEPass>();
}