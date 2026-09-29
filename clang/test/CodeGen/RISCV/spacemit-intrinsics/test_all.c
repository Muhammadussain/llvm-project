// RUN: %clang_cc1 -triple riscv64 -target-feature +zve64x -target-feature +xsmtvdot -disable-O0-optnone -emit-llvm %s -o - | opt -S -passes=mem2reg | FileCheck %s

#include <spacemit_vector.h>

// CHECK-LABEL: @test_vmadot
// CHECK: call <vscale x 2 x i32> @llvm.riscv.vmadot
vint32m1_t test_vmadot(vint32m1_t vd, vint8m1_t vs1, vint8m1_t vs2, size_t vl) {
    return __riscv_vmadot_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: @test_vmadotu
// CHECK: call <vscale x 2 x i32> @llvm.riscv.vmadotu
vint32m1_t test_vmadotu(vint32m1_t vd, vuint8m1_t vs1, vuint8m1_t vs2, size_t vl) {
    return __riscv_vmadotu_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: @test_vmadotus
// CHECK: call <vscale x 2 x i32> @llvm.riscv.vmadotus
vint32m1_t test_vmadotus(vint32m1_t vd, vuint8m1_t vs1, vint8m1_t vs2, size_t vl) {
    return __riscv_vmadotus_vv_i32m1(vd, vs1, vs2, vl);
}

// CHECK-LABEL: @test_vmadotsu
// CHECK: call <vscale x 2 x i32> @llvm.riscv.vmadotsu
vint32m1_t test_vmadotsu(vint32m1_t vd, vint8m1_t vs1, vuint8m1_t vs2, size_t vl) {
    return __riscv_vmadotsu_vv_i32m1(vd, vs1, vs2, vl);
}
