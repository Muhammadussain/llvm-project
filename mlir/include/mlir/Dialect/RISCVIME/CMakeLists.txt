add_mlir_dialect(RISCVIME riscv_ime)
add_mlir_doc(RISCVIME RISCVIMEDialect Dialects/ -gen-dialect-doc)

# Task 3.2: enable TableGen conversions for Op B
set(LLVM_TARGET_DEFINITIONS RISCVIME.td)
mlir_tablegen(RISCVIMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRRISCVIMEConversionsIncGen)