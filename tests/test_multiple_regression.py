import numpy as np
import pandas as pd
import pytest

from Day2.multiple_linear_regression import fit_and_evaluate, prepare_features


def test_prepare_features_encodes_categorical_columns():
    frame = pd.DataFrame({
        "R&D Spend": [10, 20, 30, 40, 50, 60],
        "State": ["New York", "California", "Florida", "New York", "California", "Florida"],
        "Profit": [100, 200, 300, 400, 500, 600],
    })
    X, y = prepare_features(frame)
    assert "State_New York" in X.columns or "State_Florida" in X.columns
    assert len(X) == len(y) == 6
    assert all(np.issubdtype(dtype, np.number) for dtype in X.dtypes)


def test_prepare_features_requires_profit_target():
    with pytest.raises(ValueError, match="Profit"):
        prepare_features(pd.DataFrame({"Spend": [1, 2, 3]}))


def test_fit_and_evaluate_returns_metrics_for_synthetic_data():
    rng = np.random.default_rng(7)
    spend = np.arange(1, 61, dtype=float)
    frame = pd.DataFrame({
        "R&D Spend": spend,
        "Administration": spend * 2,
        "Marketing Spend": spend * 3,
        "State": ["A", "B", "C"] * 20,
        "Profit": spend * 4 + 100,
    })
    model, X_train, X_test, y_train, y_test, predictions, metrics = fit_and_evaluate(frame)
    assert len(predictions) == len(y_test)
    assert set(["train_r2", "test_mse", "test_rmse", "test_r2", "adjusted_test_r2"]) <= set(metrics)
    assert metrics["test_rmse"] >= 0
    assert model.n_features_in_ == X_train.shape[1] == X_test.shape[1]
