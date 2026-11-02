from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_PATH = Path(__file__).resolve().parent / "50_Startups.csv"

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset not found at {DATA_PATH}. Download 50_Startups.csv "
        "and place it in the Day2 directory."
    )

dataset = pd.read_csv(DATA_PATH)

X = dataset.drop(columns=["Profit"])
y = dataset["Profit"]

# Convert the categorical State column into model-ready indicator features.
X = pd.get_dummies(X, drop_first=True, dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=1 / 3, random_state=1
)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

pred_train = regressor.predict(X_train)
pred_test = regressor.predict(X_test)

train_r2 = r2_score(y_train, pred_train)
test_r2 = r2_score(y_test, pred_test)
test_mse = mean_squared_error(y_test, pred_test)
test_rmse = np.sqrt(test_mse)

n_test = len(y_test)
feature_count = X_test.shape[1]
adjusted_test_r2 = (
    1 - (1 - test_r2) * (n_test - 1) / (n_test - feature_count - 1)
    if n_test > feature_count + 1
    else float("nan")
)

print(f"Training R²: {train_r2:.4f}")
print(f"Test MSE: {test_mse:,.2f}")
print(f"Test RMSE: {test_rmse:,.2f}")
print(f"Test R²: {test_r2:.4f}")
print(f"Adjusted test R²: {adjusted_test_r2:.4f}")
print("\nFeature coefficients:")
for feature, coefficient in zip(X.columns, regressor.coef_):
    print(f"{feature}: {coefficient:,.4f}")
print(f"Intercept: {regressor.intercept_:,.4f}")

# Actual vs predicted values on the held-out test set.
actual = y_test.to_numpy()
plt.figure(figsize=(8, 6))
plt.scatter(actual, pred_test)
limits = [
    min(actual.min(), pred_test.min()),
    max(actual.max(), pred_test.max()),
]
plt.plot(limits, limits, "r--", label="Ideal prediction")
plt.title("Actual vs Predicted Profit (Test Set)")
plt.xlabel("Actual Profit")
plt.ylabel("Predicted Profit")
plt.legend()
plt.tight_layout()
plt.show()

# Residual diagnostics.
residuals = actual - pred_test
plt.figure(figsize=(8, 6))
plt.scatter(pred_test, residuals)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted Profit")
plt.xlabel("Predicted Profit")
plt.ylabel("Residual")
plt.tight_layout()
plt.show()
