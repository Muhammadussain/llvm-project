//===- IntrinsicsRISCVXSmt.td - SpacemiT IME intrinsics ------*- tablegen -*-===//

let TargetPrefix = "riscv" in {
  defm vmadot   : RISCVTernaryWide;
  defm vmadotu  : RISCVTernaryWide;
  defm vmadotus : RISCVTernaryWide;
  defm vmadotsu : RISCVTernaryWide;
}
