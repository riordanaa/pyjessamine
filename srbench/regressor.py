"""
SRBench 2025 entry point for Jessamine symbolic regression.

Required exports (per SRBench contribution guide):
  - est:            sklearn-compatible Regressor object
  - model(est, X):  returns sympy-compatible string of the final model
  - complexity(est): returns node count of the SymPy parse tree
  - eval_kwargs:    (optional) method-specific arguments to evaluate_model.py
  - hyper_params:   list of dicts for hyperparameter tuning
"""

from pyjessamine import JessamineRegressor, model, complexity  # noqa: F401

# Default estimator for benchmark runs
est = JessamineRegressor(
    max_time=2 * 60 * 60 - 10 * 60,  # 2 hours minus 10 min overhead
    output_size=6,
    scratch_size=6,
    parameter_size=2,
    num_time_steps=3,
    max_epochs=10,
    op_inventory="rational",
    random_state=42,
    lambda_model=0.01,
    lambda_parameter=0.01,
    lambda_operand=0.01,
    stop_threshold=0.001,
    num_to_keep=20,
    num_to_generate=40,
    simplifier=True,
    verbosity=0,
)

# No hyperparameter tuning for now
hyper_params = [{}]

eval_kwargs = {
    "test_params": {
        "output_size": 3,
        "scratch_size": 3,
        "parameter_size": 1,
        "num_time_steps": 2,
        "max_epochs": 2,
        "num_to_keep": 5,
        "num_to_generate": 10,
        "max_time": 120,
        "random_state": 42,
    },
}
