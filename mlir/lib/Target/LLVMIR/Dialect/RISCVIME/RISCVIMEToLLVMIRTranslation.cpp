//===- RISCVIMEToLLVMIRTranslation.cpp - RISCVIME to LLVM IR -------------===//
//
// Task 3.1: LLVMTranslationDialectInterface for RISCVIME.
// Task 3.2: RISCVIMEConversions.inc is generated and available; however,
//           TableGen's createIntrinsicCall does not support fixed↔scalable
//           vector conversion (an MLIR limitation). We therefore provide
//           manual translation using llvm.vector.insert/extract intrinsics.
//           This is the standard approach for scalable dialects in MLIR.
//
// VLEN=256 → vscale=4. Fixed N → scalable N/4.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVMIR/Dialect/RISCVIME/RISCVIMEToLLVMIRTranslation.h"
#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"
#include "mlir/Target/LLVMIR/ModuleTranslation.h"

#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsRISCV.h"

using namespace mlir;
using namespace mlir::LLVM;

namespace {

static llvm::Value *fixedToScalable(llvm::IRBuilderBase &builder,
                                     llvm::Module *module, llvm::Value *fixed,
                                     llvm::ScalableVectorType *scalTy) {
  llvm::Value *poison = llvm::PoisonValue::get(scalTy);
  llvm::Function *intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::vector_insert,
      {scalTy, fixed->getType()});
  llvm::Value *zero = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(builder.getContext()), 0);
  return builder.CreateCall(intrinsic, {poison, fixed, zero});
}

static llvm::Value *scalableToFixed(llvm::IRBuilderBase &builder,
                                     llvm::Module *module, llvm::Value *scalable,
                                     llvm::FixedVectorType *fixedTy) {
  llvm::Function *intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::vector_extract,
      {fixedTy, scalable->getType()});
  llvm::Value *zero = llvm::ConstantInt::get(
      llvm::Type::getInt64Ty(builder.getContext()), 0);
  return builder.CreateCall(intrinsic, {scalable, zero});
}

static LogicalResult convertVmadotIntr(riscv_ime::VmadotIntrOp op,
                                        llvm::IRBuilderBase &builder,
                                        LLVM::ModuleTranslation &mt) {
  llvm::LLVMContext &ctx = builder.getContext();
  llvm::Module *module = mt.getLLVMModule();

  llvm::Value *acc = mt.lookupValue(op.getAcc());
  llvm::Value *vs1 = mt.lookupValue(op.getVs1());
  llvm::Value *vs2 = mt.lookupValue(op.getVs2());
  if (!acc || !vs1 || !vs2)
    return failure();

  auto *accFixedTy = llvm::cast<llvm::FixedVectorType>(acc->getType());
  auto *vs1FixedTy = llvm::cast<llvm::FixedVectorType>(vs1->getType());
  auto *vs2FixedTy = llvm::cast<llvm::FixedVectorType>(vs2->getType());

  auto *accScalTy = llvm::ScalableVectorType::get(
      accFixedTy->getElementType(), accFixedTy->getNumElements() / 4);
  auto *vs1ScalTy = llvm::ScalableVectorType::get(
      vs1FixedTy->getElementType(), vs1FixedTy->getNumElements() / 4);
  auto *vs2ScalTy = llvm::ScalableVectorType::get(
      vs2FixedTy->getElementType(), vs2FixedTy->getNumElements() / 4);

  llvm::Value *accS = fixedToScalable(builder, module, acc, accScalTy);
  llvm::Value *vs1S = fixedToScalable(builder, module, vs1, vs1ScalTy);
  llvm::Value *vs2S = fixedToScalable(builder, module, vs2, vs2ScalTy);

  llvm::Type *i64 = llvm::Type::getInt64Ty(ctx);
  llvm::Value *vl = llvm::ConstantInt::get(i64, 32);
  llvm::Value *policy = llvm::ConstantInt::get(i64, 2);

  llvm::Function *intrinsic = llvm::Intrinsic::getOrInsertDeclaration(
      module, llvm::Intrinsic::riscv_vmadotus,
      {accScalTy, vs1ScalTy, vs2ScalTy, i64});

  llvm::Value *result = builder.CreateCall(
      intrinsic, {accS, vs1S, vs2S, vl, policy});

  llvm::Value *resultFixed =
      scalableToFixed(builder, module, result, accFixedTy);

  mt.mapValue(op.getResult(), resultFixed);
  return success();
}

class RISCVIMEDialectLLVMIRTranslationInterface
    : public LLVMTranslationDialectInterface {
public:
  using LLVMTranslationDialectInterface::LLVMTranslationDialectInterface;

  LogicalResult
  convertOperation(Operation *op, llvm::IRBuilderBase &builder,
                   LLVM::ModuleTranslation &moduleTranslation) const final {
    if (auto vmadot = dyn_cast<riscv_ime::VmadotIntrOp>(op))
      return convertVmadotIntr(vmadot, builder, moduleTranslation);

    Operation &opInst = *op;
    #include "mlir/Dialect/RISCVIME/RISCVIMEConversions.inc"
    return failure();
  }
};

} // namespace

void mlir::riscv_ime::registerRISCVIMEDialectTranslation(
    DialectRegistry &registry) {
  registry.addExtension(
      +[](MLIRContext *ctx, riscv_ime::RISCVIMEDialect *dialect) {
        dialect->addInterface<RISCVIMEDialectLLVMIRTranslationInterface>();
      });
}