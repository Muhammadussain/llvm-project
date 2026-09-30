add_mlir_dialect_library(MLIRRISCVIMEToLLVMIRTranslation
  RISCVIMEToLLVMIRTranslation.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Target/LLVMIR/Dialect/RISCVIME

  DEPENDS
    MLIRRISCVIMEConversionsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRLLVMDialect
    MLIRTargetLLVMIRExport
    MLIRRISCVIMEDialect
)