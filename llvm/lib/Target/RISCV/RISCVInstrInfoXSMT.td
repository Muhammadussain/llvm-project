//===-- RISCVInstrInfoXSMT.td - SpacemiT IME instructions ----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file describes the vendor extensions defined by SpacemiT.
// IME (Intelligent Matrix Extension) - XSMTVdot
// Based on: https://github.com/spacemit-com/riscv-ime-extension-spec
//
//===----------------------------------------------------------------------===//

// Ensure instructions are placed in the RISCV namespace.
let Namespace = "RISCV" in {

//===----------------------------------------------------------------------===//
// Instruction Class for real instructions
//===----------------------------------------------------------------------===//

class XSMTVDot<bits<3> funct3, string opcodestr>
    : Instruction {
  bits<32> Inst;
  bits<5> vd;
  bits<5> vs1;
  bits<5> vs2;

  let OutOperandList = (outs VR:$vd);
  let InOperandList = (ins VR:$vs1, VR:$vs2);
  let AsmString = opcodestr # " $vd, $vs1, $vs2";
  let Constraints = "$vd = $vs2";

  // Encoding as per SpacemiT IME spec v1.0:
  // [31:26] funct7 = 111000
  // [25] vm = 0 (unmasked)
  // [24:20] vs2
  // [19:15] vs1
  // [14:12] funct3
  // [11:7]  vd
  // [6:0]   opcode = 0x2b (Custom-1)
  let Inst{31-26} = 0b111000;
  let Inst{25}    = 0b0;
  let Inst{24-20} = vs2;
  let Inst{19-15} = vs1;
  let Inst{14-12} = funct3;
  let Inst{11-7}  = vd;
  let Inst{6-0}   = 0b0101011;

  // DecoderNamespace for a separate decoder table.
  let DecoderNamespace = "XSMT";

  let Size = 4;
  let mayLoad = 0;
  let mayStore = 0;
  let hasSideEffects = 0;
}

//===----------------------------------------------------------------------===//
// Actual Instructions
//===----------------------------------------------------------------------===//

def SMT_VMADOT   : XSMTVDot<0b001, "smt.vmadot">;
def SMT_VMADOTU  : XSMTVDot<0b000, "smt.vmadotu">;
def SMT_VMADOTSU : XSMTVDot<0b010, "smt.vmadotsu">;
def SMT_VMADOTUS : XSMTVDot<0b011, "smt.vmadotus">;

//===----------------------------------------------------------------------===//
// Dummy pseudo-instruction to force `vmask` into OpName.
// Fields are defined to satisfy TableGen, but `isPseudo` makes it ignored for encoding.
//===----------------------------------------------------------------------===//

let isPseudo = 1, hasSideEffects = 0, mayLoad = 0, mayStore = 0 in
def DUMMY_MASK_INST : Instruction {
  bits<5> vd;
  bits<5> vs1;
  bits<5> vs2;
  bits<5> vmask;   // dummy field – not used

  let OutOperandList = (outs VR:$vd);
  let InOperandList = (ins VR:$vs1, VR:$vs2, GPR:$vmask);
  let AsmString = "dummy $vd, $vs1, $vs2, $vmask";
  let Size = 0;
}

//===----------------------------------------------------------------------===//
// DAG Patterns (temporarily disabled – to be added later)
//===----------------------------------------------------------------------===//
// def : Pat<(int_riscv_xsmt_vmadot   VR:$vd, VR:$vs1, VR:$vs2), (SMT_VMADOT   $vd, $vs1, $vs2)>;
// def : Pat<(int_riscv_xsmt_vmadotu  VR:$vd, VR:$vs1, VR:$vs2), (SMT_VMADOTU  $vd, $vs1, $vs2)>;
// def : Pat<(int_riscv_xsmt_vmadotsu VR:$vd, VR:$vs1, VR:$vs2), (SMT_VMADOTSU $vd, $vs1, $vs2)>;
// def : Pat<(int_riscv_xsmt_vmadotus VR:$vd, VR:$vs1, VR:$vs2), (SMT_VMADOTUS $vd, $vs1, $vs2)>;

} // end Namespace = "RISCV"
