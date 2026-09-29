add_mlir_dialect_library(MLIRRISCVIMEDialect
  RISCVIMEDialect.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/RISCVIME

  DEPENDS
    MLIRRISCVIMEIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRSupport
    MLIRVectorDialect
    MLIRSideEffectInterfaces
)