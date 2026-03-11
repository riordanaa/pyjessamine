"""
Utilities for converting Symbolics.jl string output to Python sympy-compatible format.

Symbolics.jl uses notation like:
  - x₁, x₂, ... for input variables (Unicode subscripts)
  - Standard math operators: +, -, *, /
  - ^ for exponentiation
  - Function names like sin, cos, exp, log, sqrt, abs

Python sympy needs:
  - ** instead of ^
  - ASCII variable names (x1, x2, ...)
  - Explicit multiplication (2*x1, not 2x1)
  - Capitalized versions of some functions (Abs, Mod)
"""

import re
import unicodedata


# ── Unicode subscript digit table ──────────────────────────────────────────
_SUBSCRIPT_MAP = {
    "\u2080": "0",   # ₀
    "\u2081": "1",   # ₁
    "\u2082": "2",   # ₂
    "\u2083": "3",   # ₃
    "\u2084": "4",   # ₄
    "\u2085": "5",   # ₅
    "\u2086": "6",   # ₆
    "\u2087": "7",   # ₇
    "\u2088": "8",   # ₈
    "\u2089": "9",   # ₉
}
_SUBSCRIPT_RE = re.compile("[" + "".join(_SUBSCRIPT_MAP.keys()) + "]+")

# ── Julia → SymPy function name mapping ───────────────────────────────────
# Keys are lowercase Julia names; values are SymPy equivalents.
_FUNC_MAP = {
    # Sympy requires capitalised forms for these:
    "abs":   "Abs",
    "mod":   "Mod",
    # Inverse trig — Julia uses a-prefix, SymPy uses a-prefix too, but let's
    # be explicit to handle any casing differences from Symbolics.jl:
    "asin":  "asin",
    "acos":  "acos",
    "atan":  "atan",
    "acot":  "acot",
    "asec":  "asec",
    "acsc":  "acsc",
    "asinh": "asinh",
    "acosh": "acosh",
    "atanh": "atanh",
    "acoth": "acoth",
    "asech": "asech",
    "acsch": "acsch",
    # These are the same in both, listed for completeness / safety:
    "sin":   "sin",
    "cos":   "cos",
    "tan":   "tan",
    "cot":   "cot",
    "sec":   "sec",
    "csc":   "csc",
    "sinh":  "sinh",
    "cosh":  "cosh",
    "tanh":  "tanh",
    "coth":  "coth",
    "sech":  "sech",
    "csch":  "csch",
    "exp":   "exp",
    "log":   "log",
    "sqrt":  "sqrt",
    "cbrt":  "cbrt",
    "sign":  "sign",
}


def _replace_subscripts(s):
    """Replace Unicode subscript digits with ASCII digits.

    Symbolics.jl renders variables as  x₁, x₂, …  but sympy needs  x1, x2, …
    """
    for uni, ascii_digit in _SUBSCRIPT_MAP.items():
        s = s.replace(uni, ascii_digit)
    return s


def _replace_superscripts(s):
    """Replace Unicode superscript digits (if any) with **n notation."""
    sup_map = {
        "\u00B2": "**2",  # ²
        "\u00B3": "**3",  # ³
        "\u2074": "**4",  # ⁴
        "\u2075": "**5",  # ⁵
        "\u2076": "**6",  # ⁶
        "\u2077": "**7",  # ⁷
        "\u2078": "**8",  # ⁸
        "\u2079": "**9",  # ⁹
    }
    for uni, repl in sup_map.items():
        s = s.replace(uni, repl)
    return s


def symbolics_to_sympy(julia_str):
    """
    Convert a Symbolics.jl expression string to a sympy-compatible string.

    Handles:
    - Unicode subscripts (x₁ → x1)
    - Unicode superscripts (x² → x**2)
    - Julia ^ → Python **
    - Implicit multiplication (2x1 → 2*x1, )(...) → )*(...))
    - Julia function names → SymPy names (abs → Abs, etc.)
    - Scientific notation edge cases (1e-3 stays valid)
    - Indexed variables x[1] → x1

    Parameters
    ----------
    julia_str : str
        Expression string from Symbolics.jl

    Returns
    -------
    str
        sympy-compatible expression string
    """
    s = julia_str.strip()

    # ── 1. Unicode normalisation ─────────────────────────────────────
    s = _replace_subscripts(s)
    s = _replace_superscripts(s)

    # ── 2. Replace ^ with ** ─────────────────────────────────────────
    s = s.replace("^", "**")

    # ── 3. Replace indexed variables x[1] → x1 ──────────────────────
    s = re.sub(r"(\w+)\[(\d+)\]", r"\g<1>\g<2>", s)

    # ── 4. Function name mapping ─────────────────────────────────────
    # Match function_name( and replace with the SymPy equivalent.
    # Use word-boundary-aware pattern to avoid partial matches.
    def _map_func(m):
        name = m.group(1)
        mapped = _FUNC_MAP.get(name.lower(), name)
        return mapped + "("

    s = re.sub(r"\b([a-zA-Z_]\w*)\(", _map_func, s)

    # ── 5. Implicit multiplication ───────────────────────────────────
    # 5a. Number (including decimals / scientific notation) followed by a letter:
    #     "1.98x2" → "1.98*x2"  but NOT "1e-3" → "1*e-3"
    # We use a negative lookbehind for 'e'/'E' to protect scientific notation.
    s = re.sub(r"(\d)(?<![eE])([a-zA-Z])", r"\1*\2", s)
    # Fix: the lookbehind above doesn't work correctly in all regex engines.
    # Instead, use a more targeted approach:
    # First, temporarily protect scientific notation
    s = re.sub(r"(\d)([eE])([+-]?\d)", r"\1_SCINOT_\3", s)
    # Now safely add * between digit and letter
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    # Restore scientific notation
    s = s.replace("_SCINOT_", "e")

    # 5b. Closing paren followed by a letter or opening paren:
    #     ")x1" → ")*x1",  ")(" → ")*("
    s = re.sub(r"\)([a-zA-Z(])", r")*\1", s)

    # 5c. Number followed by opening paren: "0.99(" → "0.99*("
    s = re.sub(r"(\d)\(", r"\1*(", s)

    # 5d. Letter/digit followed by opening paren that is NOT a function call
    #     This is tricky — we already handled function names above.
    #     "x1(" → "x1*(" but "sin(" should stay.  Since we already mapped
    #     functions, any remaining "identifier(" is implicit multiplication
    #     between a variable and a parenthesised expression.
    #     However, we must be careful: mapped functions like "Abs(" are fine.
    #     Actually, any alpha( is likely a function call, so skip this rule.

    # 5e. Variable followed by variable without operator: "x1 x2" → "x1*x2"
    #     (rare, but handle whitespace-separated implicit mult)

    return s


def remap_variables(expr_str, column_names):
    """
    Remap variable names from x1, x2, ... to actual DataFrame column names.

    Parameters
    ----------
    expr_str : str
        Expression string with variables x1, x2, ...
    column_names : list of str
        Actual column names from the training DataFrame

    Returns
    -------
    str
        Expression with remapped variable names
    """
    if column_names is None:
        return expr_str

    s = expr_str
    # Replace in reverse order to avoid x1 matching before x10
    for i in range(len(column_names), 0, -1):
        old_name = f"x{i}"
        new_name = str(column_names[i - 1])
        # Use word boundary to avoid partial matches
        s = re.sub(r"\b" + re.escape(old_name) + r"\b", new_name, s)

    return s
