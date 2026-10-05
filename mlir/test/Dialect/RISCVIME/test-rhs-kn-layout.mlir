func.func @test_kn(%A: vector<4x8xi8>, %B: vector<8x4xi8>, %C: vector<4x4xi32>) 
    -> vector<4x4xi32> {
  %0 = vector.contract {
    indexing_maps = [
      affine_map<(d0, d1, d2) -> (d0, d2)>,
      affine_map<(d0, d1, d2) -> (d2, d1)>,
      affine_map<(d0, d1, d2) -> (d0, d1)>
    ],
    iterator_types = ["parallel", "parallel", "reduction"],
    kind = #vector.kind<add>
  } %A, %B, %C : vector<4x8xi8>, vector<8x4xi8> into vector<4x4xi32>
  return %0 : vector<4x4xi32>
}
