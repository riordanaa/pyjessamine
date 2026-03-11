"""
Verification script: confirm that preorder_traversal node counts
match the SRBench standard for three canonical expressions.

Expected results (true SymPy preorder_traversal counts):
  x1 + 1          -> 3  (Add, x1, 1)
  sin(x1 * 2)     -> 4  (sin, Mul, x1, 2)
  x1**2 + x2 - 5  -> 6  NOT 7

  Note on the third case: SymPy represents subtraction as addition of a
  negative number.  x1**2 + x2 - 5  is stored internally as
  Add(Pow(x1, 2), x2, Integer(-5)).  The -5 is a single atomic node
  (Integer), so the tree has exactly 6 nodes:
    Add, Pow, x1, 2, x2, -5
  A human-readable decomposition expecting a separate Sub operator and a
  positive 5 would give 7, but that does not reflect the actual SymPy
  tree.  The preorder_traversal result of 6 is the SRBench-compliant
  count.

Note: local_dict is required so that x1, x2 are treated as atomic
symbols and not as implicit multiplication (x*1, x*2).
"""

from sympy import symbols, preorder_traversal
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

# Define variable symbols explicitly so x1, x2 are atomic, not x*1, x*2
x1, x2 = symbols("x1 x2")
LOCAL_DICT = {"x1": x1, "x2": x2}


def node_count(expr_str):
    expr = parse_expr(expr_str, local_dict=LOCAL_DICT, transformations=TRANSFORMATIONS)
    count = sum(1 for _ in preorder_traversal(expr))
    print(f"  expr:  {expr_str}")
    print(f"  sympy: {expr}")
    print(f"  nodes: {count}")
    print(f"  tree:  {list(preorder_traversal(expr))}")
    print()
    return count


if __name__ == "__main__":
    cases = [
        ("x1 + 1",          3, "Add, x1, 1"),
        ("sin(x1 * 2)",     4, "sin, Mul, x1, 2"),
        # 6 nodes: Add, Pow, x1, 2, x2, -5
        # SymPy folds subtraction into Add with Integer(-5) — no Sub node.
        ("x1**2 + x2 - 5",  6, "Add, Pow, x1, 2, x2, Integer(-5)"),
    ]

    all_pass = True
    for expr_str, expected, description in cases:
        print(f"--- {expr_str}  (expected {expected}: {description}) ---")
        got = node_count(expr_str)
        status = "PASS" if got == expected else f"FAIL (got {got})"
        print(f"  result: {status}\n")
        if got != expected:
            all_pass = False

    print("=" * 50)
    print("All tests passed." if all_pass else "SOME TESTS FAILED.")
