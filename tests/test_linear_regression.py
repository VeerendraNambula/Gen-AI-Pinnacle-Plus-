import numpy as np
import pytest

from Day1.linear_regression import MyLinearRegression


def test_gradient_descent_learns_linear_relationship():
    X = np.arange(1, 21, dtype=float).reshape(-1, 1)
    y = 2.5 * X[:, 0] + 7.0
    model = MyLinearRegression(lr=0.05, n_iters=1200).fit(X, y)

    predictions = model.predict([[5.0], [10.0]])
    assert predictions == pytest.approx([19.5, 32.0], abs=0.1)
    assert model.cost_history[-1] < model.cost_history[0]


def test_predict_requires_fitted_model():
    with pytest.raises(RuntimeError, match="Call fit"):
        MyLinearRegression().predict([[1.0]])


def test_rejects_mismatched_training_rows():
    with pytest.raises(ValueError, match="same non-zero number"):
        MyLinearRegression().fit([[1.0], [2.0]], [3.0])
