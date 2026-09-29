# RUN: llvm-mc --triple=riscv64 -mattr=+xsmtvdot %s -o - | FileCheck %s

smt.vmadot   v8, v4, v12
# CHECK: smt.vmadot   v8, v4, v12

smt.vmadotu  v8, v4, v12
# CHECK: smt.vmadotu  v8, v4, v12

smt.vmadotus v8, v4, v12
# CHECK: smt.vmadotus v8, v4, v12

smt.vmadotsu v8, v4, v12
# CHECK: smt.vmadotsu v8, v4, v12
