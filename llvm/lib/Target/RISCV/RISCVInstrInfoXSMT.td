//===-- RISCVInstrInfoXSMT.td - SpacemiT IME instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// XSMTVDot: smt.vmadot / vmadotu / vmadotsu / vmadotus  (LMUL=1 only)
//===----------------------------------------------------------------------===//

// ═══════════════════════════════════════════════════════════
// PART 1: ENCODING FORMAT
// ═══════════════════════════════════════════════════════════

class XSMTVDot<bits<3> funct3, string opcodestr>
    : RVInst<(outs VR:$vd), (ins VR:$vs1, VR:$vs2, VMaskOp:$vm),
             opcodestr, "$vd, $vs1, $vs2$vm", [], InstFormatR> {
  bits<5> vs2;
  bits<5> vs1;
  bits<5> vd;
  bit vm;

  let Inst{31-26} = 0b111000;
  let Inst{25} = vm;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = funct3;
  let Inst{11-7} = vd;
  let Inst{6-0} = 0b0101011;
  let hasSideEffects = 0;
  let mayLoad = 0;
  let mayStore = 0;
  let VMConstraint = true;
  let UseNamedOperandTable = true;
}

// ═══════════════════════════════════════════════════════════
// PART 2: REAL INSTRUCTIONS
// ═══════════════════════════════════════════════════════════

let DecoderNamespace = "XSMT" in {
let Predicates = [HasVendorXSMTVDot], Uses = [VL, VTYPE] in {
  def SMT_VMADOT_VV   : XSMTVDot<0b001, "smt.vmadot">;
  def SMT_VMADOTU_VV  : XSMTVDot<0b000, "smt.vmadotu">;
  def SMT_VMADOTSU_VV : XSMTVDot<0b010, "smt.vmadotsu">;
  def SMT_VMADOTUS_VV : XSMTVDot<0b011, "smt.vmadotus">;
}
}

// ═══════════════════════════════════════════════════════════
// PART 3: PSEUDO INSTRUCTIONS (expanded with defm)
// ═══════════════════════════════════════════════════════════

let Predicates = [HasVendorXSMTVDot] in {
  defm PseudoSMT_VMADOT : VPseudoBinaryV_VV<V_M1>,
                          SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                      "ReadVIALUV", V_M1.MX,
                                      forcePassthruRead=true>;
  defm PseudoSMT_VMADOTU : VPseudoBinaryV_VV<V_M1>,
                           SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                       "ReadVIALUV", V_M1.MX,
                                       forcePassthruRead=true>;
  defm PseudoSMT_VMADOTSU : VPseudoBinaryV_VV<V_M1>,
                            SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                        "ReadVIALUV", V_M1.MX,
                                        forcePassthruRead=true>;
  defm PseudoSMT_VMADOTUS : VPseudoBinaryV_VV<V_M1>,
                            SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                        "ReadVIALUV", V_M1.MX,
                                        forcePassthruRead=true>;
}

// ═══════════════════════════════════════════════════════════
// PART 4: PATTERN RULES (expanded from multiclass + foreach)
// ═══════════════════════════════════════════════════════════

let Predicates = [HasVendorXSMTVDot] in {

  // vmadot: int8m1 × int8m1 → int32m1
  let Predicates = GetVTypePredicates<VI32M1>.Predicates in
  defm : VPatTernaryWithPolicy<"int_riscv_vmadot", "PseudoSMT_VMADOT", "VV",
                               VI32M1.Vector, VI8M1.Vector, VI8M1.Vector,
                               VI32M1.Mask, VI32M1.Log2SEW, VI8M1.LMul,
                               VI32M1.RegClass, VI8M1.RegClass, VI8M1.RegClass>;

  // vmadotu: uint8m1 × uint8m1 → int32m1
  let Predicates = GetVTypePredicates<VI32M1>.Predicates in
  defm : VPatTernaryWithPolicy<"int_riscv_vmadotu", "PseudoSMT_VMADOTU", "VV",
                               VI32M1.Vector, VI8M1.Vector, VI8M1.Vector,
                               VI32M1.Mask, VI32M1.Log2SEW, VI8M1.LMul,
                               VI32M1.RegClass, VI8M1.RegClass, VI8M1.RegClass>;

  // vmadotsu: uint8m1 × int8m1 → int32m1
  let Predicates = GetVTypePredicates<VI32M1>.Predicates in
  defm : VPatTernaryWithPolicy<"int_riscv_vmadotsu", "PseudoSMT_VMADOTSU", "VV",
                               VI32M1.Vector, VI8M1.Vector, VI8M1.Vector,
                               VI32M1.Mask, VI32M1.Log2SEW, VI8M1.LMul,
                               VI32M1.RegClass, VI8M1.RegClass, VI8M1.RegClass>;

  // vmadotus: int8m1 × uint8m1 → int32m1
  let Predicates = GetVTypePredicates<VI32M1>.Predicates in
  defm : VPatTernaryWithPolicy<"int_riscv_vmadotus", "PseudoSMT_VMADOTUS", "VV",
                               VI32M1.Vector, VI8M1.Vector, VI8M1.Vector,
                               VI32M1.Mask, VI32M1.Log2SEW, VI8M1.LMul,
                               VI32M1.RegClass, VI8M1.RegClass, VI8M1.RegClass>;
}