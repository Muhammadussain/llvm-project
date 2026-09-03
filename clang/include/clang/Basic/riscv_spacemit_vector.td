//===----------------------------------------------------------------------===//
// Clang Builtins for RISC-V SpacemiT IME (XSMTVdot)
//===----------------------------------------------------------------------===//
//
// This file defines the builtins for RISC-V SpacemiT IME extension.
// Based on: https://github.com/spacemit-com/riscv-ime-extension-spec
//
//===----------------------------------------------------------------------===//

include "clang/Basic/riscv_vector_common.td"

//===----------------------------------------------------------------------===//
// SpacemiT IME Dot Product Extension (XSMTVdot)
//===----------------------------------------------------------------------===//

multiclass XSMTVdotBuiltin<list<list<string>> prototypes> {
  // Legal LMULs: 1, 1/2, 1/4 (Log2LMUL = 0, -1, -2)
  let RequiredFeatures = ["xsmtvdot"],
      UnMaskedPolicyScheme = HasPolicyOperand,
      HasMaskedOffOperand = false,
      Log2LMUL = [-2, -1, 0],
      OverloadedName = NAME in {
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "i", prototypes>;
  }
}

// 4 instructions with sign combinations
defm xsmt_vmadot   : XSMTVdotBuiltin<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)v"]]>;
defm xsmt_vmadotu  : XSMTVdotBuiltin<[["vv", "Uv", "UvUv(FixedSEW:8)Uv(FixedSEW:8)Uv"]]>;
defm xsmt_vmadotus : XSMTVdotBuiltin<[["vv", "v", "vv(FixedSEW:8)Uv(FixedSEW:8)v"]]>;
defm xsmt_vmadotsu : XSMTVdotBuiltin<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)Uv"]]>;

//===----------------------------------------------------------------------===//
// Prototype format:
//   "vv"        – vector-vector operation
//   "v"/"Uv"    – signed/unsigned vector result (i32/ui32)
//   "v(FixedSEW:8)v(FixedSEW:8)v" – accumulator (i32), vs1 (i8), vs2 (i8)
//   LMUL = 1, 1/2, 1/4 (spec-compliant)
//   vd must be even-numbered (hardware constraint)
//===----------------------------------------------------------------------===//
