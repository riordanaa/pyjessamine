"""
Julia bridge for Jessamine.jl via juliacall.

Handles Julia initialization, package loading, and provides Python-callable
wrappers around the Julia PythonInterface.jl functions.
"""

import os
import numpy as np

_jl = None
_initialized = False


def _get_jessamine_path():
    """Get the path to the Jessamine.jl project root."""
    # Go up from python/pyjessamine/ to the repo root
    this_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.normpath(os.path.join(this_dir, "..", ".."))
    return repo_root.replace("\\", "/")


def init_julia():
    """Initialize Julia and load Jessamine.jl. Called lazily on first use."""
    global _jl, _initialized
    if _initialized:
        return _jl

    # Set env var BEFORE importing juliacall so Jessamine skips PyCall/SymPy
    os.environ["JESSAMINE_NO_PYCALL"] = "1"

    import juliacall
    _jl = juliacall.Main

    jessamine_path = _get_jessamine_path()

    # Activate the Jessamine project and load the package
    _jl.seval("using Pkg")
    _jl.seval(f'Pkg.activate("{jessamine_path}")')
    _jl.seval("using Jessamine")

    _initialized = True
    return _jl


def fit(X, y, **kwargs):
    """
    Fit a Jessamine model.

    Parameters
    ----------
    X : numpy.ndarray of shape (n_samples, n_features)
    y : numpy.ndarray of shape (n_samples,)
    **kwargs : dict
        Keyword arguments passed to Jessamine.jessamine_fit().
        See PythonInterface.jl for valid options.

    Returns
    -------
    fit_result : Julia NamedTuple (opaque handle)
    """
    jl = init_julia()

    X_f64 = np.ascontiguousarray(X, dtype=np.float64)
    y_f64 = np.ascontiguousarray(y, dtype=np.float64)

    # Build Julia keyword arguments
    jl_kwargs = {}
    for k, v in kwargs.items():
        if v is not None:
            jl_kwargs[k] = v

    result = jl.Jessamine.jessamine_fit(X_f64, y_f64, **jl_kwargs)
    return result


def predict(fit_result, X):
    """
    Predict using a fitted Jessamine model.

    Parameters
    ----------
    fit_result : Julia NamedTuple from fit()
    X : numpy.ndarray of shape (n_samples, n_features)

    Returns
    -------
    y_pred : numpy.ndarray of shape (n_samples,)
    """
    jl = init_julia()

    X_f64 = np.ascontiguousarray(X, dtype=np.float64)
    y_pred_jl = jl.Jessamine.jessamine_predict(fit_result, X_f64)
    return np.array(y_pred_jl, dtype=np.float64)


def symbolic_string(fit_result):
    """
    Get the symbolic expression string from a fitted model.

    Parameters
    ----------
    fit_result : Julia NamedTuple from fit()

    Returns
    -------
    expr_str : str
        String representation of the symbolic expression (Symbolics.jl format).
    """
    jl = init_julia()
    return str(jl.Jessamine.jessamine_symbolic_string(fit_result))


def complexity(fit_result):
    """
    Get the complexity of the fitted model.

    Parameters
    ----------
    fit_result : Julia NamedTuple from fit()

    Returns
    -------
    complexity : int
    """
    jl = init_julia()
    return int(jl.Jessamine.jessamine_complexity(fit_result))
