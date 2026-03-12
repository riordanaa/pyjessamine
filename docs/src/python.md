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

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_time` | 300 | Maximum search time (seconds) |
| `output_size` | 6 | Output slots in the genome |
| `scratch_size` | 6 | Scratch (intermediate) slots |
| `parameter_size` | 2 | Learnable scalar parameters per genome |
| `num_time_steps` | 3 | Genome evaluation iterations |
| `max_epochs` | 10 | Maximum VNS epochs |
| `op_inventory` | "polynomial" | Operation set: "polynomial", "rational", "explog", "trig" |
| `random_state` | None | Random seed for reproducibility |
| `lambda_model` | 0.01 | Regularization for model coefficients |
| `lambda_parameter` | 0.01 | Regularization for genome parameters |
| `lambda_operand` | 0.01 | Complexity penalty weight |
| `stop_threshold` | 0.001 | Early stopping threshold |
| `num_to_keep` | 20 | Elite agents per generation |
| `num_to_generate` | 40 | New agents per generation |
| `simplifier` | true | Run simplification epoch |
| `verbosity` | 0 | 0=silent, 1=info, 2=debug |

## Operation Inventories

- **polynomial**: +, -, *, constant multiply
- **rational**: +, -, *, /, constant multiply
- **explog**: +, -, *, exp, log, constant multiply
- **trig**: +, -, *, sin, cos, constant multiply

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
