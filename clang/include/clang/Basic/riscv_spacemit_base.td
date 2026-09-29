//===----------------------------------------------------------------------===//
// Minimal base definitions for SpacemiT vector builtins
// Copied/adapted from riscv_vector.td (only essentials)
//===----------------------------------------------------------------------===//

// Policy schemes
class PolicyScheme<string name> {
  string Name = name;
}
def HasPolicyOperand : PolicyScheme<"HasPolicy">;
def NonePolicy : PolicyScheme<"None">;
def HasPassthruOperand : PolicyScheme<"HasPassthru">;

// Base multiclass for vector builtins (minimal stub)
multiclass RVVOutOp1Op2BuiltinSet<string name, string suffix, list<list<string>> types,
                                  bit hasMasked = 0, bit hasVL = 1,
                                  string OverloadedName = name,
                                  string RequiredFeatures = ""> {
  // Stub – required by TableGen backend
}

// Helper strings (needed for type parsing)
class IsFloat<string type> {
  let SupportOverloading = false;
}

// Additional classes needed for FixedSEW handling
class FixedSEW<int sew> {}
class Log2LMUL<int lmul> {}
