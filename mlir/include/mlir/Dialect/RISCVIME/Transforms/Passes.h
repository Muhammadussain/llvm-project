//===- Passes.h - RISCVIME pass declarations -----------------*- C++ -*-===//

#ifndef MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
class Pass;

namespace riscv_ime {

/// Task 2: Lower vector.contract → riscv_ime.vmadot
std::unique_ptr<::mlir::Pass> createLowerContractionToRISCVIMEPass();
void registerLowerContractionToRISCVIMEPass();

/// Task 3.1: Legalize riscv_ime.vmadot → riscv_ime.intr.vmadotus
std::unique_ptr<::mlir::Pass> createLegalizeForLLVMExportPass();
void registerLegalizeForLLVMExportPass();

} // namespace riscv_ime
} // namespace mlir

#endif
// //===- Passes.h - RISCVIME pass declarations -----------------*- C++ -*-===//

// #ifndef MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H
// #define MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H

// #include "mlir/Pass/Pass.h"

// namespace mlir {
// namespace riscv_ime {

// /// Register the vector.contract -> riscv_ime.vmadot lowering pass.
// void registerLowerContractionToRISCVIMEPass();
// std::unique_ptr<Pass> createLowerContractionToRISCVIMEPass();    // ← ADD

// void registerLegalizeForLLVMExportPass();
// std::unique_ptr<Pass> createLegalizeForLLVMExportPass();          // ← ADD


// } // namespace riscv_ime
// } // namespace mlir

// #endif // MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H