llvm.func @vmadot_llvm(%lhs: vector<32xi8>, %rhs: vector<32xi8>,
                       %acc: vector<16xi32>, %vl: i64) -> vector<16xi32> {
  %r = riscv_ime.vmadot %lhs, %rhs, %acc, %vl
       : vector<32xi8>, vector<32xi8>, vector<16xi32>, i64
       into vector<16xi32>
  llvm.return %r : vector<16xi32>
}
