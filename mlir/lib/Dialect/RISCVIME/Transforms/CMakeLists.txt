add_mlir_dialect_library(MLIRRISCVIMETransforms
  LowerContractionToRISCVIMEVMADOTPattern.cpp
  LegalizeForLLVMExport.cpp                             

  ADDITIONAL_HEADER_DIRS
    ${MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/RISCVIME/Transforms

  DEPENDS
    MLIRRISCVIMEIncGen
    MLIRRISCVIMEConversionsIncGen

  LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    MLIRTransforms
    MLIRArithDialect
    MLIRVectorDialect
    MLIRFuncDialect
    MLIRLLVMDialect
    MLIRRISCVIMEDialect
)