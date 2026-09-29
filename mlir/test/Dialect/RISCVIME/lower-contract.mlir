// RUN: mlir-opt %s --lower-contraction-to-riscv-ime | FileCheck %s

// ===== Test 1: Exact 4x8x4 (single vmadot) =====
// CHECK-LABEL: func.func @contract_4x8x4
func.func @contract_4x8x4(%lhs: vector<4x8xi8>, %rhs: vector<4x8xi8>, %acc: vector<4x4xi32>) -> vector<4x4xi32> {
  // CHECK-COUNT-1: riscv_ime.vmadot
  // CHECK-NOT: vector.contract
  %0 = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<4x8xi8>, vector<4x8xi8> into vector<4x4xi32>
  return %0 : vector<4x4xi32>
}

// ===== Test 2: 8x8x16 (8 vmadots) =====
// CHECK-LABEL: func.func @contract_8x8x16
func.func @contract_8x8x16(%lhs: vector<8x16xi8>, %rhs: vector<8x16xi8>, %acc: vector<8x8xi32>) -> vector<8x8xi32> {
  // CHECK-COUNT-4: riscv_ime.vmadot
  // CHECK-NOT: vector.contract
  %0 = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<8x16xi8>, vector<8x16xi8> into vector<8x8xi32>
  return %0 : vector<8x8xi32>
}

// ===== Test 3: 16x16x16 (16 vmadots) =====
// CHECK-LABEL: func.func @contract_16x16x16
func.func @contract_16x16x16(%lhs: vector<16x16xi8>, %rhs: vector<16x16xi8>, %acc: vector<16x16xi32>) -> vector<16x16xi32> {
  // CHECK-COUNT-16: riscv_ime.vmadot
  %0 = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<16x16xi8>, vector<16x16xi8> into vector<16x16xi32>
  return %0 : vector<16x16xi32>
}

// ===== Test 4: Wrong shape (should NOT match) =====
// CHECK-LABEL: func.func @contract_wrong
func.func @contract_wrong(%lhs: vector<5x8xi8>, %rhs: vector<5x8xi8>, %acc: vector<5x5xi32>) -> vector<5x5xi32> {
  // CHECK: vector.contract
  // CHECK-NOT: riscv_ime.vmadot
  %0 = vector.contract {
      indexing_maps = [
        affine_map<(d0, d1, d2) -> (d0, d2)>,
        affine_map<(d0, d1, d2) -> (d1, d2)>,
        affine_map<(d0, d1, d2) -> (d0, d1)>
      ],
      iterator_types = ["parallel", "parallel", "reduction"],
      kind = #vector.kind<add>
  } %lhs, %rhs, %acc : vector<5x8xi8>, vector<5x8xi8> into vector<5x5xi32>
  return %0 : vector<5x5xi32>
}