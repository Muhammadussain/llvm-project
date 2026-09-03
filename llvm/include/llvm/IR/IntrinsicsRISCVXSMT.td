//===- IntrinsicsRISCVXSMT.td - SpacemiT IME intrinsics -----*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the SpacemiT IME vendor intrinsics for RISC-V.
// Based on: https://github.com/spacemit-com/riscv-ime-extension-spec
//
//===----------------------------------------------------------------------===//

let TargetPrefix = "riscv" in {

  // vmadot: signed × signed (funct3 = 0b001)
  def int_riscv_xsmt_vmadot   : Intrinsic<[llvm_anyvector_ty],
                                          [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                                          [IntrNoMem, IntrSpeculatable]>;

  // vmadotu: unsigned × unsigned (funct3 = 0b000)
  def int_riscv_xsmt_vmadotu  : Intrinsic<[llvm_anyvector_ty],
                                          [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                                          [IntrNoMem, IntrSpeculatable]>;

  // vmadotus: unsigned × signed (funct3 = 0b011)
  def int_riscv_xsmt_vmadotus : Intrinsic<[llvm_anyvector_ty],
                                          [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                                          [IntrNoMem, IntrSpeculatable]>;

  // vmadotsu: signed × unsigned (funct3 = 0b010)
  def int_riscv_xsmt_vmadotsu : Intrinsic<[llvm_anyvector_ty],
                                          [LLVMMatchType<0>, LLVMMatchType<0>, LLVMMatchType<0>],
                                          [IntrNoMem, IntrSpeculatable]>;
}
