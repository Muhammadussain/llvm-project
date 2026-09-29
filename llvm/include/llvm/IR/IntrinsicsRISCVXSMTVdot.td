//===-- IntrinsicsRISCVXSMTVdot.td - SpacemiT XSMTVDot Intrinsics --------===//
//
// Defines LLVM IR intrinsics for SpacemiT Vector Dot Product Extension
//===----------------------------------------------------------------------===//
//===- IntrinsicsRISCVXSMTVdot.td - SpacemiT IME intrinsics --*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {
  defm vmadot   : RISCVTernaryWide;
  defm vmadotu  : RISCVTernaryWide;
  defm vmadotus : RISCVTernaryWide;
  defm vmadotsu : RISCVTernaryWide;
}

/*
let TargetPrefix = "riscv" in {
  // vmadot: return int32 = int8 * int8 + int32(acc)
  // Operand order: (acc, vs1, vs2)
  def int_riscv_xsmtvdot_vmadot : Intrinsic<[llvm_vint32_ty],
      [llvm_vint32_ty, llvm_vint8_ty, llvm_vint8_ty],
      [IntrNoMem, IntrSpeculatable]>;

  def int_riscv_xsmtvdot_vmadotu : Intrinsic<[llvm_vint32_ty],
      [llvm_vint32_ty, llvm_vuint8_ty, llvm_vuint8_ty],
      [IntrNoMem, IntrSpeculatable]>;

  def int_riscv_xsmtvdot_vmadotus : Intrinsic<[llvm_vint32_ty],
      [llvm_vint32_ty, llvm_vuint8_ty, llvm_vint8_ty],
      [IntrNoMem, IntrSpeculatable]>;

  def int_riscv_xsmtvdot_vmadotsu : Intrinsic<[llvm_vint32_ty],
      [llvm_vint32_ty, llvm_vint8_ty, llvm_vuint8_ty],
      [IntrNoMem, IntrSpeculatable]>;
}
*/
