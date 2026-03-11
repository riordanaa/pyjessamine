#!/usr/bin/env python
"""
run_evaluation.py — Evaluate JessamineRegressor on a benchmark dataset.

Runs a single experiment, prints a summary, and persists every metric that
matters for SRBench-style analysis to both CSV and JSON files.

Usage
-----
    python run_evaluation.py                          # defaults (Strogatz–Kepler)
    python run_evaluation.py --dataset polynomial     # y = x1^2 + 2*x2
    python run_evaluation.py --dataset kepler         # T^2 = a^3  (Strogatz)
    python run_evaluation.py --max-time 600 --seed 7  # override hyper-params
    python run_evaluation.py --output results.csv     # custom output path

The script saves results to ``results/eval_<dataset>_<timestamp>.{csv,json}``.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

import numpy as np
from sklearn.model_selection import train_test_split

from pyjessamine import JessamineRegressor, complexity, model


# ── Benchmark dataset generators ─────────────────────────────────────────

def _make_polynomial(n_samples=200, noise=0.0, seed=42):
    """y = x1^2 + 2*x2   (2-feature polynomial)"""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, 2))
    y = X[:, 0] ** 2 + 2.0 * X[:, 1] + noise * rng.standard_normal(n_samples)
    return X, y, "y = x1**2 + 2*x2", ["x1", "x2"]


def _make_kepler(n_samples=200, noise=0.0, seed=42):
    """Strogatz / Kepler's third law:  T^2 = a^3  →  T = a^(3/2)"""
    rng = np.random.default_rng(seed)
    a = rng.uniform(0.5, 10.0, size=n_samples)
    T = a ** 1.5 + noise * rng.standard_normal(n_samples)
    X = a.reshape(-1, 1)
    return X, T, "T = a**(3/2)", ["a"]


def _make_nguyen7(n_samples=200, noise=0.0, seed=42):
    """Nguyen-7:  y = log(x+1) + log(x^2+1)"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 2, size=n_samples)
    y = np.log(x + 1) + np.log(x ** 2 + 1) + noise * rng.standard_normal(n_samples)
    X = x.reshape(-1, 1)
    return X, y, "y = log(x+1) + log(x**2+1)", ["x"]


DATASETS = {
    "polynomial": _make_polynomial,
    "kepler": _make_kepler,
    "nguyen7": _make_nguyen7,
}


# ── Main ─────────────────────────────────────────────────────────────────

def run_experiment(args):
    # ── Generate / load data ─────────────────────────────────────────
    make_fn = DATASETS[args.dataset]
    X, y, ground_truth, feature_names = make_fn(
        n_samples=args.n_samples, noise=args.noise, seed=args.data_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.data_seed,
    )

    # ── Build estimator ──────────────────────────────────────────────
    hyperparams = dict(
        max_time=args.max_time,
        output_size=args.output_size,
        scratch_size=args.scratch_size,
        parameter_size=args.parameter_size,
        num_time_steps=args.num_time_steps,
        max_epochs=args.max_epochs,
        op_inventory=args.op_inventory,
        random_state=args.seed,
        num_to_keep=args.num_to_keep,
        num_to_generate=args.num_to_generate,
        verbosity=args.verbosity,
    )
    est = JessamineRegressor(**hyperparams)

    # ── Fit ──────────────────────────────────────────────────────────
    print(f"[*] Dataset : {args.dataset}  (ground truth: {ground_truth})")
    print(f"[*] Train/Test split : {len(y_train)} / {len(y_test)}")
    print(f"[*] Hyperparams : {hyperparams}")
    print(f"[*] Fitting …")

    t0 = time.perf_counter()
    est.fit(X_train, y_train)
    fit_seconds = time.perf_counter() - t0

    # ── Evaluate ─────────────────────────────────────────────────────
    r2_train = est.score(X_train, y_train)
    r2_test = est.score(X_test, y_test)
    model_str = model(est)
    model_complexity = complexity(est)

    # Validate the expression actually parses with SymPy
    sympy_valid = False
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
            implicit_multiplication,
            convert_xor,
        )
        transformations = standard_transformations + (
            implicit_multiplication,
            convert_xor,
        )
        parsed = parse_expr(model_str, transformations=transformations)
        sympy_valid = parsed is not None
    except Exception as exc:
        print(f"[!] SymPy parse failed: {exc}")

    # ── Print summary ────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Dataset        : {args.dataset}")
    print(f"  Ground truth   : {ground_truth}")
    print(f"  Discovered     : {model_str}")
    print(f"  Complexity     : {model_complexity}")
    print(f"  R2 (train)     : {r2_train:.6f}")
    print(f"  R2 (test)      : {r2_test:.6f}")
    print(f"  Runtime (s)    : {fit_seconds:.2f}")
    print(f"  SymPy valid    : {sympy_valid}")
    print("=" * 60)

    # ── Build results dict ───────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat()
    result = {
        "timestamp": timestamp,
        "dataset": args.dataset,
        "ground_truth": ground_truth,
        "n_samples": args.n_samples,
        "noise": args.noise,
        "data_seed": args.data_seed,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_features": X.shape[1],
        "feature_names": feature_names,
        # Model output
        "model_equation": model_str,
        "model_complexity": model_complexity,
        "sympy_valid": sympy_valid,
        # Scores
        "r2_train": float(r2_train),
        "r2_test": float(r2_test),
        "runtime_seconds": round(fit_seconds, 3),
        # Hyperparameters
        **{f"hp_{k}": v for k, v in hyperparams.items()},
    }

    # ── Save to disk ─────────────────────────────────────────────────
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)

    stem = args.output or f"eval_{args.dataset}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    stem = os.path.splitext(stem)[0]  # strip any extension the user added

    json_path = os.path.join(out_dir, f"{stem}.json")
    csv_path = os.path.join(out_dir, f"{stem}.csv")

    # JSON — full fidelity
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    # CSV — flat row, append-friendly
    import csv
    file_exists = os.path.isfile(csv_path)
    # For CSV, flatten feature_names list to a semicolon-joined string
    csv_result = {k: v for k, v in result.items()}
    csv_result["feature_names"] = ";".join(feature_names)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_result.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(csv_result)

    print(f"\n[OK] Results saved to:")
    print(f"    JSON : {json_path}")
    print(f"    CSV  : {csv_path}")

    return result


def main():
    p = argparse.ArgumentParser(
        description="Run Jessamine symbolic regression on a benchmark dataset.",
    )

    # Dataset
    p.add_argument(
        "--dataset", choices=list(DATASETS.keys()), default="polynomial",
        help="Benchmark problem to run (default: polynomial).",
    )
    p.add_argument("--n-samples", type=int, default=200, help="Number of data points.")
    p.add_argument("--noise", type=float, default=0.0, help="Additive Gaussian noise std.")
    p.add_argument("--data-seed", type=int, default=42, help="Seed for data generation.")

    # Jessamine hyperparameters
    p.add_argument("--max-time", type=int, default=120, help="Max search time in seconds.")
    p.add_argument("--output-size", type=int, default=6)
    p.add_argument("--scratch-size", type=int, default=6)
    p.add_argument("--parameter-size", type=int, default=2)
    p.add_argument("--num-time-steps", type=int, default=3)
    p.add_argument("--max-epochs", type=int, default=5)
    p.add_argument("--op-inventory", default="polynomial",
                   choices=["polynomial", "rational", "explog", "trig"])
    p.add_argument("--seed", type=int, default=42, help="Random seed for evolution.")
    p.add_argument("--num-to-keep", type=int, default=20)
    p.add_argument("--num-to-generate", type=int, default=40)
    p.add_argument("--verbosity", type=int, default=0)

    # Output
    p.add_argument("--output", type=str, default=None,
                   help="Output filename stem (without extension). "
                        "Defaults to eval_<dataset>_<timestamp>.")

    args = p.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
