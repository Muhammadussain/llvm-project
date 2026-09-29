// RUN: mlir-opt %s | mlir-opt | FileCheck %s

// CHECK-LABEL: func.func @vmadot_fixed
func.func @vmadot_fixed(%acc: vector<16xi32>,
                        %a: vector<32xi8>,
                        %b: vector<32xi8>) -> vector<16xi32> {
  // CHECK: riscv_ime.vmadot
  %0 = riscv_ime.vmadot %acc, %a, %b :
      vector<32xi8>, vector<32xi8> to vector<16xi32>
  return %0 : vector<16xi32>
}