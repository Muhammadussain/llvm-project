//===- RISCVIMEDialect.cpp - RISCVIME dialect implementation ----*- C++ -*-===//

#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.h"

// Needed because dependentDialects loads VectorDialect.
#include "mlir/Dialect/Vector/IR/VectorOps.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

using namespace mlir;
using namespace mlir::riscv_ime;

#include "mlir/Dialect/RISCVIME/RISCVIMEDialect.cpp.inc"

void RISCVIMEDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "mlir/Dialect/RISCVIME/RISCVIME.cpp.inc"
      >();
}

#define GET_OP_CLASSES
#include "mlir/Dialect/RISCVIME/RISCVIME.cpp.inc"