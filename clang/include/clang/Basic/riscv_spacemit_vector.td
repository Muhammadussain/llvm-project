//==--- riscv_spacemit_vector.td - SpacemiT IME (XSMTVDot) builtins -------===//

include "riscv_vector_common.td"

multiclass RVVVMADOT<list<list<string>> prototypes> {
  let RequiredFeatures = ["xsmtvdot"],
      UnMaskedPolicyScheme = HasPolicyOperand,
      HasMaskedOffOperand = false,
      Log2LMUL = [0],
      OverloadedName = NAME in {
    defm NAME : RVVOutOp1Op2BuiltinSet<NAME, "i", prototypes>;
  }
}

// dest vint32m1, vs1/vs2 SEW=8 LMUL=1
defm vmadot   : RVVVMADOT<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)v"]]>;    // i8  x i8  -> i32
defm vmadotu  : RVVVMADOT<[["vv", "v", "vv(FixedSEW:8)Uv(FixedSEW:8)Uv"]]>;  // u8  x u8  -> i32
defm vmadotus : RVVVMADOT<[["vv", "v", "vv(FixedSEW:8)Uv(FixedSEW:8)v"]]>;   // u8  x i8  -> i32
defm vmadotsu : RVVVMADOT<[["vv", "v", "vv(FixedSEW:8)v(FixedSEW:8)Uv"]]>;   // i8  x u8  -> i32
