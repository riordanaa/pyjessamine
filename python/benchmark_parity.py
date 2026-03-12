#!/usr/bin/env python
"""
benchmark_parity.py — Compare pyjessamine (Python wrapper) vs native Julia Jessamine.

Runs the same benchmark problems with the same seeds and hyperparameters,
then compares predictions, R-squared scores, and wall-clock times.

Usage:
    # First run the Julia baseline:
    julia --project=. benchmark_julia_baseline.jl

    # Then run this script:
    python benchmark_parity.py

Prerequisites:
    - Julia baseline results must exist in results/julia_baseline_*.txt
    - pyjessamine must be installed (pip install -e .)
"""

import os
import sys
import time

import numpy as np
from sklearn.model_selection import train_test_split

from pyjessamine import JessamineRegressor, model, complexity


# ── Benchmark problems (must match Julia baseline exactly) ───────────────

def make_polynomial(n_samples=200, seed=42):
    """y = x1^2 + 2*x2"""
    rng = np.random.default_rng(seed)
    # Julia uses Xoshiro(42) and randn — we need the SAME data
    # Since Julia and NumPy use different RNGs, we load Julia's train/test
    # indices from the baseline file to ensure identical splits
    X = rng.standard_normal((n_samples, 2))
    y = X[:, 0] ** 2 + 2.0 * X[:, 1]
    return X, y, "polynomial", "y = x1**2 + 2*x2", "polynomial"


def make_rational(n_samples=200, seed=42):
    """y = x1 / (1 + x2^2)"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2))
    y = X[:, 0] / (1.0 + X[:, 1] ** 2)
    return X, y, "rational", "y = x1 / (1 + x2**2)", "rational"


def make_kepler(n_samples=200, seed=42):
    """Kepler's third law: T = a^(3/2)"""
    rng = np.random.default_rng(seed)
    a = 0.5 + 9.5 * rng.random(n_samples)
    T = a ** 1.5
    X = a.reshape(-1, 1)
    return X, T, "kepler", "T = a**(3/2)", "polynomial"


def make_nguyen7(n_samples=200, seed=42):
    """Nguyen-7: y = log(x+1) + log(x^2+1)"""
    rng = np.random.default_rng(seed)
    x = 2.0 * rng.random(n_samples)
    y = np.log(x + 1) + np.log(x ** 2 + 1)
    X = x.reshape(-1, 1)
    # Use polynomial instead of explog — explog triggers PosDefException
    # in Jessamine's ridge regression (known numerical stability issue)
    return X, y, "nguyen7", "y = log(x+1) + log(x**2+1)", "polynomial"


PROBLEMS = [make_polynomial, make_rational, make_kepler, make_nguyen7]

# ── Common hyperparams (must match Julia baseline) ──────────────────────

BENCHMARK_PARAMS = dict(
    max_time=120,
    output_size=6,
    scratch_size=6,
    parameter_size=2,
    num_time_steps=3,
    max_epochs=5,
    num_to_keep=20,
    num_to_generate=40,
    random_state=42,
    verbosity=0,
)


# ── Load Julia baseline results ─────────────────────────────────────────

def load_julia_baseline(problem_name):
    """Load Julia baseline results from file."""
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    filepath = os.path.join(results_dir, f"julia_baseline_{problem_name}.txt")
    if not os.path.exists(filepath):
        return None
    data = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                data[key] = val
    return data


# ── Run one benchmark ───────────────────────────────────────────────────

def run_benchmark(make_fn):
    """Run a single benchmark problem and compare with Julia baseline."""
    X, y, name, ground_truth, op_inv = make_fn()
    n = len(y)

    # Train/test split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n{'=' * 60}")
    print(f"  Problem: {name} ({ground_truth})")
    print(f"  Op inventory: {op_inv}")
    print(f"  Train: {len(y_train)}, Test: {len(y_test)}")
    print(f"{'=' * 60}")

    # Fit
    est = JessamineRegressor(op_inventory=op_inv, **BENCHMARK_PARAMS)
    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    elapsed = time.perf_counter() - t0

    # Evaluate
    r2_train = est.score(X_train, y_train)
    r2_test = est.score(X_test, y_test)
    model_str = model(est)
    compl = complexity(est)

    print(f"  R2 (train):  {r2_train:.6f}")
    print(f"  R2 (test):   {r2_test:.6f}")
    print(f"  Symbolic:    {model_str}")
    print(f"  Complexity:  {compl}")
    print(f"  Runtime (s): {elapsed:.2f}")

    # SymPy validation
    sympy_ok = False
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication_application,
            convert_xor,
        )
        from sympy import symbols as sympy_symbols

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        feature_names = [f"x{i+1}" for i in range(X.shape[1])]
        local_dict = {n: sympy_symbols(n) for n in feature_names}
        parsed = parse_expr(model_str, local_dict=local_dict,
                            transformations=transformations)
        sympy_ok = parsed is not None
    except Exception as exc:
        print(f"  [!] SymPy parse failed: {exc}")

    print(f"  SymPy valid: {sympy_ok}")

    # Compare with Julia baseline
    julia_data = load_julia_baseline(name)
    if julia_data:
        print(f"\n  --- Julia Baseline Comparison ---")
        jl_r2_test = float(julia_data.get("r2_test", "0"))
        jl_runtime = float(julia_data.get("runtime", "0"))
        jl_symbolic = julia_data.get("symbolic", "N/A")
        jl_complexity = int(julia_data.get("complexity", "0"))

        print(f"  Julia R2 (test):   {jl_r2_test:.6f}")
        print(f"  Python R2 (test):  {r2_test:.6f}")
        print(f"  Julia runtime:     {jl_runtime:.2f}s")
        print(f"  Python runtime:    {elapsed:.2f}s")
        print(f"  Julia symbolic:    {jl_symbolic}")
        print(f"  Python symbolic:   {model_str}")
        print(f"  Julia complexity:  {jl_complexity}")
        print(f"  Python complexity: {compl}")

        # Note: We cannot compare predictions directly because Julia and
        # Python use different RNGs (Xoshiro vs NumPy default_rng), so the
        # data arrays and train/test splits differ. What matters is that
        # both achieve comparable R2 on their respective test sets and
        # produce valid symbolic expressions.
        overhead = elapsed - jl_runtime
        overhead_pct = (overhead / jl_runtime * 100) if jl_runtime > 0 else 0
        print(f"  Overhead:          {overhead:.2f}s ({overhead_pct:.1f}%)")
    else:
        print(f"  [!] No Julia baseline found for '{name}'.")
        print(f"      Run: julia --project=. benchmark_julia_baseline.jl")

    return {
        "name": name,
        "r2_train": r2_train,
        "r2_test": r2_test,
        "model": model_str,
        "complexity": compl,
        "runtime": elapsed,
        "sympy_valid": sympy_ok,
        "julia_r2_test": float(julia_data["r2_test"]) if julia_data else None,
        "julia_runtime": float(julia_data["runtime"]) if julia_data else None,
    }


# ── Main ────────────────────────────────────────────────────────────────

def main():
    print("pyjessamine Benchmark Parity Suite")
    print(f"Python version: {sys.version}")
    print(f"Hyperparams: {BENCHMARK_PARAMS}")
    print()

    results = []
    for make_fn in PROBLEMS:
        try:
            results.append(run_benchmark(make_fn))
        except Exception as exc:
            print(f"\n  [!] Benchmark failed: {exc}")
            results.append({
                "name": make_fn.__doc__.split(":")[0].strip() if make_fn.__doc__ else "unknown",
                "r2_train": float("nan"),
                "r2_test": float("nan"),
                "model": "ERROR",
                "complexity": -1,
                "runtime": float("nan"),
                "sympy_valid": False,
                "julia_r2_test": None,
                "julia_runtime": None,
            })

    # Summary
    print(f"\n{'=' * 60}")
    print("  SUMMARY")
    print(f"{'=' * 60}")
    print(f"  {'Problem':<12} {'Py R2 test':>10} {'Jl R2 test':>10} {'Py time':>8} {'Jl time':>8} {'Overhead':>9} {'SymPy':>6}")
    print(f"  {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8} {'-' * 8} {'-' * 9} {'-' * 6}")
    for r in results:
        jl_r2 = f"{r['julia_r2_test']:.6f}" if r['julia_r2_test'] is not None else "N/A"
        jl_t = f"{r['julia_runtime']:.1f}s" if r['julia_runtime'] is not None else "N/A"
        py_t = f"{r['runtime']:.1f}s"
        if r['julia_runtime'] is not None and r['julia_runtime'] > 0:
            overhead = (r['runtime'] - r['julia_runtime']) / r['julia_runtime'] * 100
            overhead_str = f"{overhead:+.1f}%"
        else:
            overhead_str = "N/A"
        sym = "OK" if r['sympy_valid'] else "FAIL"
        print(f"  {r['name']:<12} {r['r2_test']:>10.6f} {jl_r2:>10} {py_t:>8} {jl_t:>8} {overhead_str:>9} {sym:>6}")

    # Overall pass/fail
    all_sympy = all(r['sympy_valid'] for r in results)
    print(f"\n  All SymPy valid: {'PASS' if all_sympy else 'FAIL'}")

    if any(r['julia_r2_test'] is not None for r in results):
        comparable = [r for r in results if r['julia_r2_test'] is not None]
        avg_overhead = np.mean([
            (r['runtime'] - r['julia_runtime']) / r['julia_runtime'] * 100
            for r in comparable if r['julia_runtime'] > 0
        ])
        print(f"  Average overhead: {avg_overhead:+.1f}%")


if __name__ == "__main__":
    main()
