"""
Tests for pyjessamine wrapper.
Run with: python -m pytest tests/test_regressor.py -v
"""

import numpy as np
import pytest


# Quick-test parameters for fast local runs
QUICK_PARAMS = dict(
    output_size=3,
    scratch_size=3,
    parameter_size=1,
    num_time_steps=2,
    max_epochs=2,
    num_to_keep=5,
    num_to_generate=10,
    max_time=120,
    random_state=42,
    verbosity=0,
)


@pytest.fixture(scope="module")
def polynomial_data():
    """Generate polynomial test data: y = x1^2 + 2*x2"""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((100, 2))
    y = X[:, 0] ** 2 + 2 * X[:, 1]
    return X, y


@pytest.fixture(scope="module")
def fitted_model(polynomial_data):
    """Fit a JessamineRegressor on polynomial data (shared across tests)."""
    from pyjessamine import JessamineRegressor

    X, y = polynomial_data
    est = JessamineRegressor(**QUICK_PARAMS)
    est.fit(X, y)
    return est


class TestJessamineRegressor:
    def test_fit_returns_self(self, polynomial_data):
        from pyjessamine import JessamineRegressor

        X, y = polynomial_data
        est = JessamineRegressor(**QUICK_PARAMS)
        result = est.fit(X, y)
        assert result is est

    def test_predict_shape(self, fitted_model, polynomial_data):
        X, y = polynomial_data
        y_pred = fitted_model.predict(X)
        assert y_pred.shape == (X.shape[0],)

    def test_predict_finite(self, fitted_model, polynomial_data):
        X, _ = polynomial_data
        y_pred = fitted_model.predict(X)
        assert np.all(np.isfinite(y_pred))

    def test_model_string(self, fitted_model):
        from pyjessamine import model

        expr_str = model(fitted_model)
        assert isinstance(expr_str, str)
        assert len(expr_str) > 0
        assert "ERROR" not in expr_str

    def test_complexity(self, fitted_model):
        from pyjessamine import complexity

        c = complexity(fitted_model)
        assert isinstance(c, int)
        assert c > 0

    def test_sympy_parse(self, fitted_model):
        from pyjessamine import model

        expr_str = model(fitted_model)
        try:
            import sympy
            from sympy.parsing.sympy_parser import (
                parse_expr,
                standard_transformations,
                implicit_multiplication,
                convert_xor,
            )
        except ImportError:
            pytest.skip("sympy not installed")

        # parse_expr with implicit multiplication as a safety net
        transformations = standard_transformations + (
            implicit_multiplication,
            convert_xor,
        )
        expr = parse_expr(expr_str, transformations=transformations)
        assert expr is not None
        # The expression must contain at least one symbol (x1, x2, ...)
        assert len(expr.free_symbols) > 0, (
            f"Expected symbols in expression, got none: {expr_str}"
        )

    def test_sklearn_clone(self, polynomial_data):
        """Test that sklearn clone works (requires proper get_params/set_params)."""
        from sklearn.base import clone

        from pyjessamine import JessamineRegressor

        est = JessamineRegressor(**QUICK_PARAMS)
        est2 = clone(est)
        assert est.get_params() == est2.get_params()

    def test_feature_names(self, fitted_model):
        assert hasattr(fitted_model, "feature_names_")
        assert fitted_model.feature_names_ == ["x1", "x2"]

    def test_n_features(self, fitted_model):
        assert fitted_model.n_features_in_ == 2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
