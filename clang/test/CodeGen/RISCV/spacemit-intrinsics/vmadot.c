// RUN: %clang_cc1 -triple riscv64 -target-feature +zve64x \
// RUN:   -target-feature +xsmtvdot -disable-O0-optnone \
// RUN:   -emit-llvm %s -o - | opt -S -passes=mem2reg | \
// RUN:   FileCheck --check-prefix=CHECK-RV64 %s

#include <spacemit_vector.h>

// CHECK-RV64-LABEL: @test_vmadot
// CHECK-RV64: call <vscale x 2 x i32> @llvm.riscv.vmadot.nxv2i32.nxv8i8.nxv8i8.i64(
vint32m1_t test_vmadot(vint32m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  return __riscv_vmadot_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-RV64-LABEL: @test_vmadotu
// CHECK-RV64: call <vscale x 2 x i32> @llvm.riscv.vmadotu.nxv2i32.nxv8i8.nxv8i8.i64(
vint32m1_t test_vmadotu(vint32m1_t vd, vuint8m1_t vs1, vuint8m1_t vs2, size_t vl) {
  return __riscv_vmadotu_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-RV64-LABEL: @test_vmadotus
// CHECK-RV64: call <vscale x 2 x i32> @llvm.riscv.vmadotus.nxv2i32.nxv8i8.nxv8i8.i64(
vint32m1_t test_vmadotus(vint32m1_t vd, vuint8m1_t vs1, vint8m1_t vs2, size_t vl) {
  return __riscv_vmadotus_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-RV64-LABEL: @test_vmadotsu
// CHECK-RV64: call <vscale x 2 x i32> @llvm.riscv.vmadotsu.nxv2i32.nxv8i8.nxv8i8.i64(
vint32m1_t test_vmadotsu(vint32m1_t vd, vint8m1_t vs1, vuint8m1_t vs2, size_t vl) {
  return __riscv_vmadotsu_vv_i32m1(vd, vs1, vs2, vl);
}
