# LLVM/MLIR — SpacemiT IME (XSMT) Support

## Branches

| Branch | Description |
|---|---|
| `phase1-xsmt-patches` | LLVM backend: smt.vmadotus |
| `phase3-riscvime-dialect` | MLIR riscv_ime dialect + translation |

## Build Instructions

### Clone
git clone https://github.com/Muhammadussain/llvm-project.git
cd llvm-project
git checkout phase3-riscvime-dialect

### Build
export LLVM_SRC=$(pwd)
export LLVM_BUILD=$HOME/llvm-xsmt-build

rm -rf $LLVM_BUILD
mkdir -p $LLVM_BUILD
cd $LLVM_BUILD

cmake -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DLLVM_ENABLE_PROJECTS="clang;lld;mlir" \
  -DLLVM_TARGETS_TO_BUILD="RISCV;X86" \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_INSTALL_PREFIX=$HOME/llvm-xsmt-install \
  $LLVM_SRC/llvm

cmake --build . --target clang llc opt FileCheck count not mlir-opt mlir-translate llvm-objdump -j$(nproc)
cmake --install .
