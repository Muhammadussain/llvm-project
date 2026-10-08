//===-- RISCVInstrInfoXSMT.td - SpacemiT IME instructions ----*- tablegen -*-===//
//
// Native tile 4x4x8: acc = 16 x i32 (m2), src = 32 x i8 (m1).
// Widening ternary op (VPseudoTernaryW_VV / VPatTernaryW_VV).
//
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
  def SMT_VMADOT_VV   : XSMTVDot<0b011, "smt.vmadot">;
  def SMT_VMADOTU_VV  : XSMTVDot<0b000, "smt.vmadotu">;
  def SMT_VMADOTSU_VV : XSMTVDot<0b010, "smt.vmadotsu">;
  def SMT_VMADOTUS_VV : XSMTVDot<0b001, "smt.vmadotus">;
}
}

// ═══════════════════════════════════════════════════════════
// PART 3: PSEUDO INSTRUCTIONS — widening (W) with V_M1 source
//   vd = VRM2 (acc, 16 i32),  vs = VR (srcs, 32 i8)
// ═══════════════════════════════════════════════════════════

let Predicates = [HasVendorXSMTVDot] in {
  defm PseudoSMT_VMADOT : VPseudoTernaryW_VV<V_M1>,
                          SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                      "ReadVIALUV", V_M1.MX,
                                      forcePassthruRead=true>;
  defm PseudoSMT_VMADOTU : VPseudoTernaryW_VV<V_M1>,
                           SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                       "ReadVIALUV", V_M1.MX,
                                       forcePassthruRead=true>;
  defm PseudoSMT_VMADOTSU : VPseudoTernaryW_VV<V_M1>,
                            SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                        "ReadVIALUV", V_M1.MX,
                                        forcePassthruRead=true>;
  defm PseudoSMT_VMADOTUS : VPseudoTernaryW_VV<V_M1>,
                            SchedBinary<"WriteVIALUV", "ReadVIALUV",
                                        "ReadVIALUV", V_M1.MX,
                                        forcePassthruRead=true>;
}

// ═══════════════════════════════════════════════════════════
// PART 4: PATTERN RULES — VPatTernaryW_VV with custom
//          VTypeInfoToWide<VI8M1, VI32M2> (4x widening)
// ═══════════════════════════════════════════════════════════

let Predicates = [HasVendorXSMTVDot] in {

  defm : VPatTernaryW_VV<"int_riscv_vmadot", "PseudoSMT_VMADOT",
                         [VTypeInfoToWide<VI8M1, VI32M2>]>;

  defm : VPatTernaryW_VV<"int_riscv_vmadotu", "PseudoSMT_VMADOTU",
                         [VTypeInfoToWide<VI8M1, VI32M2>]>;

  defm : VPatTernaryW_VV<"int_riscv_vmadotsu", "PseudoSMT_VMADOTSU",
                         [VTypeInfoToWide<VI8M1, VI32M2>]>;

  defm : VPatTernaryW_VV<"int_riscv_vmadotus", "PseudoSMT_VMADOTUS",
                         [VTypeInfoToWide<VI8M1, VI32M2>]>;
}
