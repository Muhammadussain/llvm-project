add_mlir_dialect_library(MLIRRISCVIMETransforms
  LowerContractionToRISCVIMEVMADOTPattern.cpp

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/RISCVIME/Transforms

  DEPENDS
    MLIRRISCVIMEIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRArithDialect
    MLIRVectorDialect
    MLIRFuncDialect
    MLIRRISCVIMEDialect
)