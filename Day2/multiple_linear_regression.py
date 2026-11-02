"""Multiple linear regression for the 50 Startups learning dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "50_Startups.csv"


def prepare_features(dataset: pd.DataFrame):
    if "Profit" not in dataset.columns:
        raise ValueError("Dataset must contain a 'Profit' target column.")
    X = dataset.drop(columns=["Profit"])
    y = pd.to_numeric(dataset["Profit"], errors="coerce")
    valid = y.notna()
    X, y = X.loc[valid].copy(), y.loc[valid]
    for column in X.select_dtypes(include="object").columns:
        X[column] = X[column].astype(str).str.strip()
    X = pd.get_dummies(X, drop_first=True, dtype=int)
    return X, y


def fit_and_evaluate(dataset: pd.DataFrame, test_size=1 / 3, random_state=1):
    X, y = prepare_features(dataset)
    if len(X) < 5:
        raise ValueError("At least five valid rows are required for a train/test split.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression().fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    test_r2 = r2_score(y_test, test_predictions)
    n_test, feature_count = X_test.shape
    adjusted_r2 = (
        1 - (1 - test_r2) * (n_test - 1) / (n_test - feature_count - 1)
        if n_test > feature_count + 1
        else float("nan")
    )
    metrics = {
        "train_r2": r2_score(y_train, train_predictions),
        "test_mse": mean_squared_error(y_test, test_predictions),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, test_predictions))),
        "test_r2": test_r2,
        "adjusted_test_r2": adjusted_r2,
    }
    return model, X_train, X_test, y_train, y_test, test_predictions, metrics


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download 50_Startups.csv "
            "and place it in the Day2 directory."
        )

    dataset = pd.read_csv(DATA_PATH)
    model, X_train, X_test, y_train, y_test, predictions, metrics = fit_and_evaluate(dataset)

    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value:,.4f}")
    print("\nFeature coefficients:")
    for feature, coefficient in zip(X_train.columns, model.coef_):
        print(f"{feature}: {coefficient:,.4f}")
    print(f"Intercept: {model.intercept_:,.4f}")

    actual = y_test.to_numpy()
    plt.figure(figsize=(8, 6))
    plt.scatter(actual, predictions)
    limits = [min(actual.min(), predictions.min()), max(actual.max(), predictions.max())]
    plt.plot(limits, limits, "r--", label="Ideal prediction")
    plt.title("Actual vs Predicted Profit (Test Set)")
    plt.xlabel("Actual Profit")
    plt.ylabel("Predicted Profit")
    plt.legend()
    plt.tight_layout()
    plt.show()

    residuals = actual - predictions
    plt.figure(figsize=(8, 6))
    plt.scatter(predictions, residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals vs Predicted Profit")
    plt.xlabel("Predicted Profit")
    plt.ylabel("Residual")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
