//===- RISCVIMEToLLVMIRTranslation.h - RISCVIME to LLVM IR ------*- C++ -*-===//

#ifndef MLIR_TARGET_LLVMIR_DIALECT_RISCVIME_RISCVIMETOLLVMIRTRANSLATION_H
#define MLIR_TARGET_LLVMIR_DIALECT_RISCVIME_RISCVIMETOLLVMIRTRANSLATION_H

namespace mlir {
class DialectRegistry;

namespace riscv_ime {
/// Register translation from RISCVIME dialect to LLVM IR.
void registerRISCVIMEDialectTranslation(DialectRegistry &registry);
} // namespace riscv_ime
} // namespace mlir

#endif