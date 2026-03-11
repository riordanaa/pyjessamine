# Jessamine.jl

Julia package for evolutionary symbolic regression using static-single-assignment form expressions.

**pyjessamine** is the scikit-learn compatible Python wrapper, built for [SRBench](https://github.com/cavalab/srbench) compliance. See the full project documentation below.

---

## 1. Executive Summary

**pyjessamine** is a scikit-learn compatible Python wrapper for [Jessamine.jl](https://github.com/riordanaa/pyjessamine), a symbolic regression engine written in Julia. Jessamine evolves mathematical expressions using linear genetic programming in Static Single Assignment (SSA) form, combined with ridge regression for coefficient fitting and Variable Neighborhood Search (VNS) as the metaheuristic optimizer.

The goal of this project was to make Jessamine accessible from Python as a drop-in scikit-learn estimator, enabling it to participate in standardized benchmarks like [SRBench](https://github.com/cavalab/srbench) alongside other symbolic regression methods (PySR, gplearn, etc.). The wrapper exposes the full `fit()` / `predict()` / `score()` API, produces SymPy-parseable symbolic expressions, and includes an evaluation pipeline for running reproducible experiments on high-performance clusters.

**Key deliverables:**

- A Python package (`pyjessamine`) with `JessamineRegressor`, `model()`, and `complexity()`.
- A Julia-to-SymPy expression converter that handles all of Symbolics.jl's output formats.
- An SRBench-compliant integration (`srbench/`).
- A configurable evaluation script (`run_evaluation.py`) with CSV/JSON result logging.
- A critical bug fix in the original Jessamine.jl source code.

---

## 2. Core Julia Modifications (The Bug Fix)

### The Problem

The original Jessamine.jl codebase contained a bug in `src/MachineLayer.jl` at line 158. The function `_MGR_f` (the objective function used during parameter optimization) called a function named `run_genome_to_end`:

```julia
last_round = run_genome_to_end(c.g_spec, c.genome, genome_parameter, c.xs)
```

This function **does not exist** anywhere in the codebase. It is not defined in `GenomeCore.jl`, not exported, and not available in any module. This caused an `UndefVarError` at runtime whenever the `MachineLayer` code path was exercised (specifically during `machine_grow_and_rate`, which is called during the evolutionary search).

### The Fix

The correct function is `run_genome`, which is defined and exported from `GenomeCore.jl`. However, `run_genome` returns a `Vector` of outputs for **every time step**, not just the final one. Since `_MGR_f` only needs the output from the last time step (to feed into the ridge regression model), the fix wraps the call with `last()`:

```julia
last_round = last(run_genome(c.g_spec, c.genome, genome_parameter, c.xs))
```

This extracts the final time-step output from the full trajectory, which is exactly what the downstream code (`extend_if_singleton`, `namedtuple`, `machine_init`) expects.

### Why This Matters

Without this fix, the entire `MachineLayer` code path is non-functional. Since this is the layer responsible for combining genome-evolved features with a linear model (ridge regression), it is a core component of Jessamine's architecture. The Python wrapper cannot operate at all without this fix.

---

## 3. The Bridge Architecture

The system consists of three layers that connect Python's scikit-learn API to Julia's Jessamine engine:

```
Python (user code)              Python (bridge)                 Julia
--------------------------      --------------------------      --------------------------
JessamineRegressor.fit(X, y)    julia_bridge.fit(X, y)          Jessamine.jessamine_fit(X, y)
    |                               |                               |
    v                               v                               v
JessamineRegressor.predict(X)   julia_bridge.predict(result, X) Jessamine.jessamine_predict(...)
    |                               |                               |
    v                               v                               v
model(est)                      julia_bridge.symbolic_string()  Jessamine.jessamine_symbolic_string(...)
    |
    v
sympy_utils.symbolics_to_sympy()  -->  SymPy-compatible string
```

### Layer 1: `src/PythonInterface.jl` (Julia Side)

This module provides four top-level functions that serve as the entry points from Python:

- **`jessamine_fit(X, y; kwargs...)`** -- Accepts a matrix `X` and vector `y`, constructs the full Jessamine configuration (genome spec, solver spec, operation inventory), runs the evolutionary search, and returns a `NamedTuple` containing the best agent, genome spec, and all metadata needed for prediction.

- **`jessamine_predict(fit_result, X)`** -- Takes the fit result and a new feature matrix, evaluates the evolved genome on the new data, applies the fitted linear model, and returns predictions.

- **`jessamine_symbolic_string(fit_result)`** -- Uses `Symbolics.jl` (not SymPy/PyCall) to produce a human-readable mathematical expression. This avoids the PyCall/juliacall conflict entirely.

- **`jessamine_complexity(fit_result)`** -- Returns the total complexity of the model (number of active operands + instructions in the genome).

A critical design decision: the interface sets `JESSAMINE_NO_PYCALL=1` to prevent Julia from loading PyCall, which would conflict with juliacall. Symbolic output uses pure Julia's `Symbolics.jl` instead of the SymPy extension.

### Layer 2: `python/pyjessamine/julia_bridge.py` (Python-Julia Bridge)

This module handles all Julia initialization and data marshaling:

- **Lazy initialization**: Julia is only started on the first call to `fit()`, `predict()`, etc. This avoids slow startup if the user just imports the package.
- **Project activation**: Automatically locates the Jessamine.jl project root (relative to the package location) and activates it via `Pkg.activate()`.
- **Data conversion**: NumPy arrays are converted to contiguous `float64` arrays before passing to Julia. juliacall handles the actual memory sharing, so there is no data copying for large arrays.
- **Environment isolation**: Sets `JESSAMINE_NO_PYCALL=1` before importing juliacall, ensuring Julia never tries to load PyCall alongside juliacall.

### Layer 3: `python/pyjessamine/regressor.py` (Scikit-Learn API)

This module provides the public API:

- **`JessamineRegressor`** -- A full scikit-learn estimator inheriting from `BaseEstimator` and `RegressorMixin`. It exposes all of Jessamine's hyperparameters as constructor arguments (`max_time`, `output_size`, `scratch_size`, `parameter_size`, `num_time_steps`, `max_epochs`, `op_inventory`, etc.) and implements `fit()`, `predict()`, `score()`, `get_params()`, and `set_params()`.

- **`model(est)`** -- Returns the symbolic expression as a SymPy-compatible string. If the estimator was trained with named features (via a pandas DataFrame), the variable names in the expression are remapped to match.

- **`complexity(est)`** -- Returns the SRBench-standard complexity of the fitted model: the number of nodes in the SymPy expression parse tree, computed via `preorder_traversal`. Feature names are passed as a `local_dict` so that identifiers like `x1`, `x2` are treated as atomic symbols rather than implicit multiplication (`x*1`, `x*2`). Falls back to Julia-side genome complexity if SymPy parsing fails.

---

## 4. SRBench and SymPy Compliance

### The Challenge

SRBench requires that every symbolic regression method produce a model expression that can be parsed by `sympy.parse_expr()`. Jessamine uses Julia's `Symbolics.jl` library for symbolic output, which produces expressions in Julia notation. Several formatting differences make these strings unparseable by SymPy out of the box.

### Specific Issues and Solutions (in `sympy_utils.py`)

| Issue | Example (Symbolics.jl output) | Solution |
|-------|-------------------------------|----------|
| **Unicode subscript digits** | `x` + subscript 1, `x` + subscript 2 | Map all Unicode subscript characters (U+2080 through U+2089) to ASCII digits |
| **Unicode superscripts** | `x` + superscript 2 | Convert to `**n` notation (e.g., `**2`) |
| **Exponentiation operator** | `x^2` | Replace `^` with `**` |
| **Implicit multiplication (number-variable)** | `1.98x2` | Insert `*` between digit and letter: `1.98*x2` |
| **Implicit multiplication (paren-variable)** | `)x1` | Insert `*`: `)*x1` |
| **Implicit multiplication (number-paren)** | `0.99(` | Insert `*`: `0.99*(` |
| **Scientific notation preservation** | `1e-3` | Temporarily protect `e` in scientific notation before inserting multiplication operators, then restore |
| **Julia function names** | `abs(`, `mod(` | Map to SymPy equivalents: `Abs(`, `Mod(` |
| **Indexed variables** | `x[1]` | Convert to `x1` |

### Complexity Metric: SRBench Node-Count Standard

`complexity(est)` follows the SRBench 2.0 definition: the total number of nodes in the SymPy expression parse tree, obtained via `sympy.preorder_traversal`. Every operator, variable, and constant is one node. Verified against three canonical cases:

| Expression | Expected | Got | Notes |
|---|---|---|---|
| `x1 + 1` | 3 | 3 | Add, x1, 1 |
| `sin(x1 * 2)` | 4 | 4 | sin, Mul, x1, 2 |
| `x1**2 + x2 - 5` | 6 | 6 | SymPy folds `-5` into a single `Integer(-5)` atom; no separate Sub node |

The third case illustrates an important subtlety: SymPy represents `a - 5` as `Add(a, Integer(-5))`, so the negative constant is one node, not two. This is the canonical SRBench-compliant count.

A `local_dict` mapping feature names to SymPy symbols is passed to `parse_expr` to prevent `x1` from being treated as `x * 1` under `implicit_multiplication_application`.

### Defense in Depth

The test suite validates SymPy compliance using `parse_expr` with the `implicit_multiplication` and `convert_xor` transformations enabled as a safety net. The test asserts both that parsing succeeds and that the resulting expression contains at least one free symbol (preventing degenerate constant-only expressions from silently passing).

---

## 5. Evaluation Pipeline

### `python/run_evaluation.py`

This script is designed as a template for running reproducible symbolic regression experiments, particularly on HPC clusters where results must be logged to files rather than read from a terminal.

### Built-In Datasets

| Name | Formula | Features |
|------|---------|----------|
| `polynomial` | y = x1^2 + 2*x2 | 2 |
| `kepler` | T = a^(3/2) (Strogatz / Kepler's Third Law) | 1 |
| `nguyen7` | y = log(x+1) + log(x^2+1) | 1 |

### CLI Interface

All hyperparameters are exposed as command-line arguments:

```bash
python run_evaluation.py \
    --dataset kepler \
    --max-time 600 \
    --max-epochs 10 \
    --op-inventory explog \
    --seed 42 \
    --output kepler_run1
```

### Output Format

Every run saves two files to `python/results/`:

**JSON** (full fidelity, nested structures preserved):
```json
{
  "timestamp": "2026-03-11T20:31:26.588277+00:00",
  "dataset": "polynomial",
  "ground_truth": "y = x1**2 + 2*x2",
  "model_equation": "0.0066 + 1.994*x2 + 0.992*(x1**2)",
  "model_complexity": 14,
  "sympy_valid": true,
  "r2_train": 0.999975,
  "r2_test": 0.999965,
  "runtime_seconds": 134.25,
  "hp_max_time": 60,
  "hp_output_size": 6,
  ...
}
```

**CSV** (flat, append-friendly for aggregating multiple runs):
Each run appends one row. Column headers are written only on the first run. This makes it straightforward to aggregate results across seeds, datasets, or hyperparameter sweeps using pandas or any spreadsheet tool.

### Recorded Fields

- **Metadata**: timestamp, dataset name, ground truth formula, sample sizes, feature names
- **Model output**: discovered equation (SymPy-valid string), complexity, SymPy validation flag
- **Performance**: R-squared on train and test splits, wall-clock runtime in seconds
- **Hyperparameters**: all Jessamine hyperparameters, prefixed with `hp_` for easy filtering

This structure is designed to be directly compatible with SRBench's results analysis pipeline and supports parameter sweeps on SLURM or PBS-based clusters.

---

## 6. Test Results

All tests have been run and pass. The raw outputs are committed to the repository for verification.

### Python Unit Tests (9/9 Passed)

Full output: [`test_results.txt`](test_results.txt)

| Test | Status |
|------|--------|
| `test_fit_returns_self` | PASSED |
| `test_predict_shape` | PASSED |
| `test_predict_finite` | PASSED |
| `test_model_string` | PASSED |
| `test_complexity` | PASSED |
| `test_sympy_parse` | PASSED |
| `test_sklearn_clone` | PASSED |
| `test_feature_names` | PASSED |
| `test_n_features` | PASSED |

### Julia Interface Test

Full output: [`julia_test_results.txt`](julia_test_results.txt)

- **Rating**: 0.0799 (lower is better)
- **Predictions**: 100 samples generated successfully
- **Discovered formula**: `0.005 + 1.990*x2 + 0.995*(x1^2)` (ground truth: `x1^2 + 2*x2`)
- **Complexity**: 10

### Evaluation Pipeline Run

Full outputs: [`python/results/test_run.json`](python/results/test_run.json) | [`python/results/test_run.csv`](python/results/test_run.csv)

- **R2 (train)**: 0.999975
- **R2 (test)**: 0.999965
- **SymPy valid**: true
- **Runtime**: 134.25 seconds
- **Discovered equation**: `0.0066 + 1.994*x2 + 0.992*(x1**2)`

---

## 7. SRBench Contribution Compliance

All requirements from the [SRBench Contribution Guide](https://github.com/cavalab/srbench/blob/master/CONTRIBUTING.md) have been verified and met.

### Estimator Requirements

| Requirement | Status | Implementation |
|---|---|---|
| Open-source Python implementation | DONE | MIT-licensed, hosted at [github.com/riordanaa/pyjessamine](https://github.com/riordanaa/pyjessamine) |
| Scikit-learn compatible API | DONE | Inherits `BaseEstimator` + `RegressorMixin`; implements `fit()`, `predict()`, `score()`, `get_params()`, `set_params()` |
| `random_state` attribute | DONE | Renamed from `random_seed` to `random_state` per SRBench convention |
| `max_time` parameter | DONE | Controls Julia evolutionary search time limit in seconds |
| SIGALRM handling | DONE | Catches `signal.SIGALRM` in `fit()` on Unix to respect SRBench time enforcement |

### Submission Files

| File | Requirement | Status |
|---|---|---|
| `srbench/metadata.yml` | Authors, description, URL | DONE |
| `srbench/install.sh` | Pull from stable repo, not local source | DONE -- clones from GitHub, installs via pip, precompiles Julia |
| `srbench/regressor.py` | Exports `est`, `model()`, `complexity()`, `hyper_params`, `eval_kwargs` | DONE |

### Model Output Requirements

| Requirement | Status | Implementation |
|---|---|---|
| `model(est, X)` returns sympy-compatible string | DONE | `sympy_utils.py` converts Symbolics.jl output (Unicode subscripts, `^`, implicit multiplication) |
| Variable names match `X.columns` | DONE | DataFrame columns captured before `check_X_y` validation; `model()` remaps `x1, x2, ...` to column names |
| Operators available in SymPy | DONE | Julia `abs` mapped to `Abs`, `mod` to `Mod`, etc. |
| `complexity(est)` returns node count | DONE | Uses `sympy.preorder_traversal` (SRBench 2.0 standard); `local_dict` prevents `x1` misparse |

### Install Script Requirements

| Requirement | Status | Implementation |
|---|---|---|
| No sudo required | DONE | All installs use pip into the conda environment |
| No source code included | DONE | `install.sh` clones from GitHub at runtime |
| Uses `$CONDA_PREFIX` | DONE | Install target respects conda environment |
| Precompilation | DONE | Julia packages precompiled during install to avoid first-run latency |
