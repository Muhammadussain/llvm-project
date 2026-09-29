; RUN: llc < %s -mtriple=riscv64 -mattr=+v,+xsmtvdot -verify-machineinstrs | FileCheck %s

define <vscale x 4 x i32> @test_vmadot(<vscale x 4 x i32> %vd, <vscale x 4 x i8> %vs1, <vscale x 4 x i8> %vs2) {
; CHECK-LABEL: test_vmadot:
; CHECK: smt.vmadot
  %call = call <vscale x 4 x i32> @llvm.riscv.xsmt.vmadot.nxv4i32.nxv4i8(<vscale x 4 x i32> %vd, <vscale x 4 x i8> %vs1, <vscale x 4 x i8> %vs2, i32 4, i1 -1)
  ret <vscale x 4 x i32> %call
}
declare <vscale x 4 x i32> @llvm.riscv.xsmt.vmadot.nxv4i32.nxv4i8(<vscale x 4 x i32>, <vscale x 4 x i8>, <vscale x 4 x i8>, i32, i1)

; Add similar tests for vmadotu, vmadotus, vmadotsu
