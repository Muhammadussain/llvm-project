//===- Passes.h - RISCVIME pass declarations -----------------*- C++ -*-===//

#ifndef MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace riscv_ime {

/// Register the vector.contract -> riscv_ime.vmadot lowering pass.
void registerLowerContractionToRISCVIMEPass();

} // namespace riscv_ime
} // namespace mlir

#endif // MLIR_DIALECT_RISCVIME_TRANSFORMS_PASSES_H