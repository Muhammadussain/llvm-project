// LMUL=2 test — 4×4×8 vmadot
func.func @test_vmadot_lmul2(%acc: vector<16xi32>, 
                              %vs1: vector<32xi8>, 
                              %vs2: vector<32xi8>) -> vector<16xi32> {
  %0 = riscv_ime.vmadot %acc, %vs1, %vs2 
       : vector<32xi8>, vector<32xi8> to vector<16xi32>
  return %0 : vector<16xi32>
}
