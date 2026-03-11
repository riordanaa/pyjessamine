"""
pyjessamine - Python scikit-learn wrapper for Jessamine.jl symbolic regression.
"""

from .regressor import JessamineRegressor, model, complexity

__all__ = ["JessamineRegressor", "model", "complexity"]
__version__ = "0.1.0"
