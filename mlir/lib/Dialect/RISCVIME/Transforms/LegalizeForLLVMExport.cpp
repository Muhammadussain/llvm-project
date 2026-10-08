//===- LegalizeForLLVMExport.cpp - Legalize RISCVIME for LLVM export ------===//
//
// Task 3.1: riscv_ime.vmadot → riscv_ime.intr.vmadotus (same fixed types).
// Type conversion (fixed→scalable) happens at LLVM IR level in the
// translation interface.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Dialect/RISCVIME/Transforms/Passes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace mlir::riscv_ime;

namespace {

struct LegalizeVmadotToIntr : public OpRewritePattern<VmadotOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(VmadotOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    Value vl = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(32));
    Value policy = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(2));

    auto intr = rewriter.create<VmadotIntrOp>(
        loc, op.getResult().getType(),
        op.getAcc(), op.getVs1(), op.getVs2(), vl, policy);

    rewriter.replaceOp(op, intr.getResult());
    return success();
  }
};

struct LegalizeForLLVMExportPass
    : public PassWrapper<LegalizeForLLVMExportPass,
                         OperationPass<func::FuncOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LegalizeForLLVMExportPass)

  StringRef getArgument() const final { return "legalize-riscv-ime-for-llvm"; }
  StringRef getDescription() const final {
    return "Legalize RISCVIME ops for LLVM IR export";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<riscv_ime::RISCVIMEDialect>();
  }

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<LegalizeVmadotToIntr>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> 
mlir::riscv_ime::createLegalizeForLLVMExportPass() {
  return std::make_unique<LegalizeForLLVMExportPass>();
}

void mlir::riscv_ime::registerLegalizeForLLVMExportPass() {
  PassRegistration<LegalizeForLLVMExportPass>();
}
