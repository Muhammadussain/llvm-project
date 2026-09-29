//===-- RISCVIME.td - RISC-V IME dialect --------------------*- tablegen -*-===//
//
// SpacemiT Integrated Matrix Extension (IME) dialect.
//
//===----------------------------------------------------------------------===//

#ifndef RISCVIME_OPS
#define RISCVIME_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/IR/CommonTypeConstraints.td"

//===----------------------------------------------------------------------===//
// Dialect
//===----------------------------------------------------------------------===//

def RISCVIME_Dialect : Dialect {
  let name = "riscv_ime";
  let cppNamespace = "::mlir::riscv_ime";
  let useStrictPropertiesInAssemblyFormat = 1;
  let summary = "SpacemiT Integrated Matrix Extension (IME)";
  let description = [{
    Vendor extension for RISC-V integer matrix multiply-accumulate.
    
  }];
  let dependentDialects = [
    "vector::VectorDialect"
   
  ];
}

//===----------------------------------------------------------------------===//
// Type helper — fixed-length 1D vector
//===----------------------------------------------------------------------===//

class RISCVIMEVectorOfLength<int length, Type elementType>
    : ShapedContainerType<
        [elementType],
        And<[IsVectorOfShape<[length]>, IsFixedVectorOfAnyRankTypePred]>,
        "a vector with length " # length,
        "::mlir::VectorType">;

//===----------------------------------------------------------------------===//
// Op: riscv_ime.vmadot
//
// Atom at VLEN=256, SEW=8:
//   acc : 4x4 i32  -> 16 x i32
//   vs1 : 4x8 i8   -> 32 x i8
//   vs2 : 4x8 i8   -> 32 x i8
//
//===----------------------------------------------------------------------===//

def RISCVIME_VmadotOp : Op<RISCVIME_Dialect, "vmadot",
    [Pure,
     AllTypesMatch<["vs1", "vs2"]>,
     AllTypesMatch<["acc", "res"]>]> {
  let summary = "SpacemiT IME signed i8 x i8 -> i32 matrix MAC";
  let description = [{
    Computes acc += vs1 * vs2^T at the 4x4x8 atom level.
  }];

  let arguments = (ins
      RISCVIMEVectorOfLength<16, I32>:$acc,
      RISCVIMEVectorOfLength<32, I8>:$vs1,
      RISCVIMEVectorOfLength<32, I8>:$vs2);
  let results = (outs
      RISCVIMEVectorOfLength<16, I32>:$res);

  let assemblyFormat =
    "$acc `,` $vs1 `,` $vs2 attr-dict `:` type($vs1) `,` type($vs2) `to` type($res)";
}

#endif // RISCVIME_OPS