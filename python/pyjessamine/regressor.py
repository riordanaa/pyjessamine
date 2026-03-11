"""
Scikit-learn compatible wrapper for Jessamine.jl symbolic regression.
"""

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sympy import preorder_traversal
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, implicit_multiplication_application, convert_xor

from . import julia_bridge
from .sympy_utils import symbolics_to_sympy, remap_variables

_SYMPY_TRANSFORMATIONS = standard_transformations + (implicit_multiplication_application, convert_xor)


class JessamineRegressor(BaseEstimator, RegressorMixin):
    """
    Scikit-learn compatible symbolic regression estimator backed by Jessamine.jl.

    Jessamine evolves mathematical expressions using linear genetic programming
    in SSA form, combined with ridge regression for coefficient fitting and
    Variable Neighborhood Search (VNS) as the metaheuristic.

    Parameters
    ----------
    max_time : int, default=300
        Maximum time in seconds for the evolutionary search.
    output_size : int, default=6
        Number of output slots in the genome.
    scratch_size : int, default=6
        Number of scratch (intermediate) slots in the genome.
    parameter_size : int, default=2
        Number of learnable scalar parameters per genome.
    num_time_steps : int, default=3
        Number of genome evaluation iterations (controls expression depth).
    max_epochs : int, default=10
        Maximum number of VNS epochs.
    op_inventory : str, default="polynomial"
        Operation inventory. One of "polynomial", "rational", "explog", "trig".
    random_seed : int or None, default=None
        Random seed for reproducibility. None uses default RNG.
    lambda_model : float, default=0.01
        Regularization weight for the linear model coefficients.
    lambda_parameter : float, default=0.01
        Regularization weight for genome parameters.
    lambda_operand : float, default=0.01
        Regularization weight for operand count (complexity penalty).
    stop_threshold : float or None, default=0.001
        Stop if best rating falls below this threshold.
    num_to_keep : int, default=20
        Number of elite agents to keep per generation.
    num_to_generate : int, default=40
        Number of new agents to generate per generation.
    simplifier : bool, default=True
        Whether to run a simplification epoch after evolution.
    verbosity : int, default=0
        Verbosity level (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        max_time=300,
        output_size=6,
        scratch_size=6,
        parameter_size=2,
        num_time_steps=3,
        max_epochs=10,
        op_inventory="polynomial",
        random_seed=None,
        lambda_model=0.01,
        lambda_parameter=0.01,
        lambda_operand=0.01,
        stop_threshold=0.001,
        num_to_keep=20,
        num_to_generate=40,
        simplifier=True,
        verbosity=0,
    ):
        self.max_time = max_time
        self.output_size = output_size
        self.scratch_size = scratch_size
        self.parameter_size = parameter_size
        self.num_time_steps = num_time_steps
        self.max_epochs = max_epochs
        self.op_inventory = op_inventory
        self.random_seed = random_seed
        self.lambda_model = lambda_model
        self.lambda_parameter = lambda_parameter
        self.lambda_operand = lambda_operand
        self.stop_threshold = stop_threshold
        self.num_to_keep = num_to_keep
        self.num_to_generate = num_to_generate
        self.simplifier = simplifier
        self.verbosity = verbosity

    def fit(self, X, y):
        """
        Fit the Jessamine symbolic regression model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training input data.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self
        """
        X, y = check_X_y(X, y)

        # Store feature names if X is a DataFrame
        if hasattr(X, "columns"):
            self.feature_names_ = list(X.columns)
        else:
            self.feature_names_ = [f"x{i+1}" for i in range(X.shape[1])]

        self.n_features_in_ = X.shape[1]

        # Build kwargs for Julia
        kwargs = {
            "max_time": int(self.max_time),
            "output_size": int(self.output_size),
            "scratch_size": int(self.scratch_size),
            "parameter_size": int(self.parameter_size),
            "num_time_steps": int(self.num_time_steps),
            "max_epochs": int(self.max_epochs),
            "op_inventory": str(self.op_inventory),
            "lambda_model": float(self.lambda_model),
            "lambda_parameter": float(self.lambda_parameter),
            "lambda_operand": float(self.lambda_operand),
            "num_to_keep": int(self.num_to_keep),
            "num_to_generate": int(self.num_to_generate),
            "simplifier": bool(self.simplifier),
            "verbosity": int(self.verbosity),
        }

        if self.random_seed is not None:
            kwargs["random_seed"] = int(self.random_seed)
        if self.stop_threshold is not None:
            kwargs["stop_threshold"] = float(self.stop_threshold)

        self._fit_result = julia_bridge.fit(X, y, **kwargs)
        self.is_fitted_ = True

        return self

    def predict(self, X):
        """
        Predict using the fitted Jessamine model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "is_fitted_")
        X = check_array(X)
        return julia_bridge.predict(self._fit_result, X)


def model(est, X=None):
    """
    Get the sympy-compatible symbolic expression string from a fitted estimator.

    Parameters
    ----------
    est : JessamineRegressor
        A fitted JessamineRegressor instance.
    X : pandas.DataFrame or None, default=None
        If provided, variable names in the expression are remapped to match
        the DataFrame column names.

    Returns
    -------
    str
        A sympy-compatible expression string.
    """
    check_is_fitted(est, "is_fitted_")
    raw_str = julia_bridge.symbolic_string(est._fit_result)
    sympy_str = symbolics_to_sympy(raw_str)

    # Remap variable names if column names available
    if X is not None and hasattr(X, "columns"):
        sympy_str = remap_variables(sympy_str, list(X.columns))
    elif hasattr(est, "feature_names_"):
        sympy_str = remap_variables(sympy_str, est.feature_names_)

    return sympy_str


def complexity(est):
    """
    Get the complexity of the fitted model as the number of nodes in the
    SymPy expression parse tree (SRBench standard).

    This follows the SRBench 2.0 definition: complexity is the total number
    of nodes in the SymPy expression tree, counting all operators, variables,
    and constants. For example, ``x1 + 2`` has 3 nodes: Add, x1, 2.

    Parameters
    ----------
    est : JessamineRegressor
        A fitted JessamineRegressor instance.

    Returns
    -------
    int
        Number of nodes in the SymPy parse tree.
    """
    check_is_fitted(est, "is_fitted_")
    try:
        from sympy import symbols as sympy_symbols
        expr_str = model(est)
        # Build a local_dict so feature names like x1, x2 are treated as
        # atomic symbols rather than implicit multiplication (x*1, x*2).
        feature_names = getattr(est, "feature_names_", [])
        local_dict = {name: sympy_symbols(name) for name in feature_names}
        expr = parse_expr(expr_str, local_dict=local_dict,
                          transformations=_SYMPY_TRANSFORMATIONS)
        return sum(1 for _ in preorder_traversal(expr))
    except Exception:
        # Fallback to Julia-side genome complexity if SymPy parsing fails
        return julia_bridge.complexity(est._fit_result)
