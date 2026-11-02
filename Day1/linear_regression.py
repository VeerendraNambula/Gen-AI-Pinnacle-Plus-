"""Linear regression from scratch on the Salary_Data learning dataset."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "Salary_Data.csv"


class MyLinearRegression:
    """Batch gradient-descent linear regression with feature/target scaling."""

    def __init__(self, lr: float = 0.05, n_iters: int = 1500):
        if lr <= 0 or n_iters < 1:
            raise ValueError("lr and n_iters must be positive.")
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.cost_history = []
        self.x_mean = None
        self.x_scale = None
        self.y_mean = None
        self.y_scale = None

    @staticmethod
    def compute_cost(y, y_pred, sample_count):
        return float(np.sum((y_pred - y) ** 2) / (2 * sample_count))

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2 or len(X) != len(y) or len(y) == 0:
            raise ValueError("X must be 2D and contain the same non-zero number of rows as y.")

        self.x_mean = X.mean(axis=0)
        self.x_scale = X.std(axis=0)
        self.x_scale[self.x_scale == 0] = 1.0
        self.y_mean = float(y.mean())
        self.y_scale = float(y.std()) or 1.0

        X_scaled = (X - self.x_mean) / self.x_scale
        y_scaled = (y - self.y_mean) / self.y_scale
        sample_count, feature_count = X_scaled.shape
        self.weights = np.zeros(feature_count)
        self.bias = 0.0
        self.cost_history = []

        for _ in range(self.n_iters):
            predictions = X_scaled @ self.weights + self.bias
            self.cost_history.append(self.compute_cost(y_scaled, predictions, sample_count))
            error = predictions - y_scaled
            self.weights -= self.lr * (X_scaled.T @ error) / sample_count
            self.bias -= self.lr * float(error.mean())
        return self

    def predict(self, X):
        if self.weights is None:
            raise RuntimeError("Call fit before predict.")
        X = np.asarray(X, dtype=float)
        X_scaled = (X - self.x_mean) / self.x_scale
        scaled_predictions = X_scaled @ self.weights + self.bias
        return scaled_predictions * self.y_scale + self.y_mean


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. Download Salary_Data.csv "
            "and place it in the Day1 directory."
        )

    salary_df = pd.read_csv(DATA_PATH)
    required = {"YearsExperience", "Salary"}
    missing = required.difference(salary_df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")

    data = salary_df[list(required)].apply(pd.to_numeric, errors="coerce").dropna()
    X = data[["YearsExperience"]]
    y = data["Salary"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = MyLinearRegression().fit(X_train, y_train)
    train_predictions = model.predict(X_train)
    test_predictions = model.predict(X_test)

    for label, actual, predicted in (
        ("Train", y_train, train_predictions),
        ("Test", y_test, test_predictions),
    ):
        print(
            f"{label}: MAE={mean_absolute_error(actual, predicted):,.2f} | "
            f"RMSE={np.sqrt(mean_squared_error(actual, predicted)):,.2f} | "
            f"R²={r2_score(actual, predicted):.4f}"
        )

    print("Learned weights (scaled feature space):", model.weights)
    print("Learned bias (scaled target space):", model.bias)
    print(f"Predicted salary for 5 years: {model.predict([[5]])[0]:,.2f}")

    order = np.argsort(X_train["YearsExperience"].to_numpy())
    plt.figure(figsize=(8, 6))
    plt.scatter(X_train["YearsExperience"], y_train, marker="x", label="Training data")
    plt.scatter(X_test["YearsExperience"], y_test, marker="o", label="Test data")
    plt.plot(
        X_train["YearsExperience"].to_numpy()[order],
        train_predictions[order],
        linewidth=2,
        label="Fitted line",
    )
    plt.title("Salary vs Experience — Linear Regression")
    plt.xlabel("Years of Experience")
    plt.ylabel("Salary")
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(model.cost_history)
    plt.title("Gradient Descent Cost History")
    plt.xlabel("Iteration")
    plt.ylabel("Scaled Mean Squared Error / 2")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
