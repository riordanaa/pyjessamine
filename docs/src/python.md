# pyjessamine -- Python Wrapper

## Overview

**pyjessamine** is a scikit-learn compatible Python wrapper for Jessamine.jl. It allows Python users to use Jessamine's evolutionary symbolic regression engine through the standard `fit()` / `predict()` / `score()` API, and produces SymPy-compatible symbolic expressions.

All computation runs in Julia via `juliacall` -- the Python layer adds no performance overhead.

## Installation

### Prerequisites

- Julia 1.11+ (https://julialang.org/downloads/)
- Python 3.10+ with pip

### Steps

```bash
git clone https://github.com/riordanaa/pyjessamine.git
cd pyjessamine/python
pip install -e .
```

Julia dependencies are installed automatically on first use.

## Quick Start

```python
import numpy as np
from pyjessamine import JessamineRegressor, model, complexity

# Example: discover y = x1^2 + 2*x2
X = np.random.randn(200, 2)
y = X[:, 0]**2 + 2 * X[:, 1]

est = JessamineRegressor(max_time=120)
est.fit(X, y)

print(model(est))       # SymPy-compatible expression
print(complexity(est))  # Node count
print(est.score(X, y))  # R-squared
```

## Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_time` | `int` | 300 | Maximum search time (seconds) |
| `output_size` | `int` | 6 | Output slots in the genome |
| `scratch_size` | `int` | 6 | Scratch (intermediate) slots |
| `parameter_size` | `int` | 2 | Learnable scalar parameters per genome |
| `num_time_steps` | `int` | 3 | Genome evaluation iterations (controls expression depth) |
| `max_epochs` | `int` | 10 | Maximum VNS epochs |
| `op_inventory` | `str` | "polynomial" | Operation set: "polynomial", "rational", "explog", "trig" |
| `random_state` | `int or None` | None | Random seed for reproducibility |
| `lambda_model` | `float` | 0.01 | Regularization for model coefficients |
| `lambda_parameter` | `float` | 0.01 | Regularization for genome parameters |
| `lambda_operand` | `float` | 0.01 | Complexity penalty weight |
| `stop_threshold` | `float or None` | 0.001 | Early stopping threshold |
| `num_to_keep` | `int` | 20 | Elite agents per generation |
| `num_to_generate` | `int` | 40 | New agents per generation |
| `simplifier` | `bool` | True | Run simplification epoch |
| `verbosity` | `int` | 0 | 0=silent, 1=info, 2=debug |

## Operation Inventories

- **polynomial**: +, -, *, constant multiply
- **rational**: +, -, *, /, constant multiply
- **explog**: +, -, *, exp, log, constant multiply
- **trig**: +, -, *, sin, cos, constant multiply

## API Reference

### `JessamineRegressor`

scikit-learn compatible estimator class (`BaseEstimator`, `RegressorMixin`).

#### `JessamineRegressor.fit(X, y)`

Fit the symbolic regression model.

| Argument | Type | Description |
|----------|------|-------------|
| `X` | `array-like of shape (n_samples, n_features)` | Training input data. Accepts NumPy arrays or pandas DataFrames. |
| `y` | `array-like of shape (n_samples,)` | Target values. |
| **Returns** | `JessamineRegressor` | The fitted estimator (for method chaining). |

#### `JessamineRegressor.predict(X)`

Predict target values for new data.

| Argument | Type | Description |
|----------|------|-------------|
| `X` | `array-like of shape (n_samples, n_features)` | Input data. |
| **Returns** | `numpy.ndarray of shape (n_samples,)` | Predicted values. |

#### `JessamineRegressor.score(X, y)`

Return the R-squared score (inherited from scikit-learn `RegressorMixin`).

| Argument | Type | Description |
|----------|------|-------------|
| `X` | `array-like of shape (n_samples, n_features)` | Test input data. |
| `y` | `array-like of shape (n_samples,)` | True target values. |
| **Returns** | `float` | R-squared score (1.0 = perfect fit). |

---

### `model(est, X=None)`

Get the discovered symbolic expression as a SymPy-compatible string.

| Argument | Type | Description |
|----------|------|-------------|
| `est` | `JessamineRegressor` | A fitted estimator. |
| `X` | `pandas.DataFrame or None` | If provided, variable names are remapped to match column names (SRBench requirement). |
| **Returns** | `str` | SymPy-parseable expression string (e.g. `"0.99*x1**2 + 2.0*x2"`). |

---

### `complexity(est)`

Get the complexity of the fitted model using SRBench 2.0 node counting.

Complexity is the total number of nodes in the SymPy expression tree
(operators + variables + constants). For example, `x1 + 2` has 3 nodes.

| Argument | Type | Description |
|----------|------|-------------|
| `est` | `JessamineRegressor` | A fitted estimator. |
| **Returns** | `int` | Number of nodes in the SymPy parse tree. |

---

## Architecture

The wrapper uses a three-layer bridge:

1. **PythonInterface.jl** (Julia) -- Flat-array entry points (no MLJ Tables from Python)
2. **julia_bridge.py** (Python) -- juliacall initialization and data marshaling
3. **regressor.py** (Python) -- scikit-learn BaseEstimator + RegressorMixin

## SRBench Integration

pyjessamine includes SRBench submission files in the `srbench/` directory:

- `metadata.yml` -- Authors and description
- `install.sh` -- Installation script
- `regressor.py` -- SRBench entry point

## Running Tests

```bash
# Python tests
cd python && python -m pytest tests/test_regressor.py -v

# Julia interface test
julia --project=. test_python_interface.jl
```
