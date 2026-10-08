add_mlir_dialect(RISCVIME riscv_ime)
# Task 3.2: enable TableGen conversions for Op B
set(LLVM_TARGET_DEFINITIONS RISCVIME.td)
mlir_tablegen(RISCVIMEConversions.inc -gen-llvmir-conversions)
add_public_tablegen_target(MLIRRISCVIMEConversionsIncGen)