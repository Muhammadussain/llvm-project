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
include "mlir/Dialect/LLVMIR/LLVMOpBase.td"    


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
    "vector::VectorDialect",
    "LLVM::LLVMDialect" 
   
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
     AllTypesMatch<["acc", "res"]>]> {
  let summary = "SpacemiT IME signed i8 x i8 -> i32 matrix MAC";
  let arguments = (ins
      RISCVIMEVectorOfLength<16, I32>:$acc,    // 4x2 = 8 i32 (m1)
      RISCVIMEVectorOfLength<32, I8>:$vs1,    // 4x8 = 32 i8 (m1)
      RISCVIMEVectorOfLength<32, I8>:$vs2);   // 2x8 = 16 i8
  let results = (outs
      RISCVIMEVectorOfLength<16, I32>:$res);

  let assemblyFormat =
    "$acc `,` $vs1 `,` $vs2 attr-dict `:` type($vs1) `,` type($vs2) `to` type($res)";
}
//===----------------------------------------------------------------------===//
// Base class for LLVM intrinsic wrapper ops
//
// LLVM_IntrOpBase arg order:
//   dialect, opName, enumName,
//   overloadedResults, overloadedOperands, traits, numResults
//===----------------------------------------------------------------------===//

class RISCVIME_IntrOp<string mnemonic, list<int> overloadedResults,
                      list<int> overloadedOperands, int numResults,
                      list<Trait> traits = []>
    : LLVM_IntrOpBase</*dialect=*/RISCVIME_Dialect,
                      /*opName=*/"intr." # mnemonic,
                      /*enumName=*/"riscv_" # !subst(".", "_", mnemonic),
                      /*overloadedResults=*/overloadedResults,
                      /*overloadedOperands=*/overloadedOperands,
                      /*traits=*/traits,
                      /*numResults=*/numResults>;

//===----------------------------------------------------------------------===//
// Op B: riscv_ime.intr.vmadotus  (LLVM intrinsic wrapper, SCALABLE)
//
// Matches Phase-1 intrinsic:
//   @llvm.riscv.vmadotus.nxv4i32.nxv8i8.nxv8i8(acc,vs1,vs2,vl,policy)
// (at VLEN=256, vscale=4)
//===----------------------------------------------------------------------===//

def RISCVIME_VmadotIntrOp
    : RISCVIME_IntrOp<"vmadotus",
                      /*overloadedResults=*/[0],
                      /*overloadedOperands=*/[1, 2, 3],      
                      /*numResults=*/1,
                      /*traits=*/[Pure]> {
  let summary = "LLVM intrinsic wrapper for @llvm.riscv.vmadotus";
  let arguments = (ins
      LLVM_AnyVector:$acc,
      LLVM_AnyVector:$vs1,
      LLVM_AnyVector:$vs2,
      I64:$vl,
      I64:$policy);
  let results = (outs LLVM_AnyVector:$res);
}

#endif // RISCVIME_OPS