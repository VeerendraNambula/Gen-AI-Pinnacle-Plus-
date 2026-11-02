# Day 2: Multiple Linear Regression

Build a multiple linear regression model to estimate startup profit from research and development spend, administration spend, marketing spend, and state.

## Objective

- Prepare numeric and categorical predictors for regression.
- Fit a multiple linear regression model with scikit-learn.
- Compare predictions with held-out test values.
- Evaluate performance with MSE, RMSE, R², and adjusted R².
- Inspect actual-versus-predicted values and residuals.

## Dataset

The script expects `50_Startups.csv` in this directory.

- Features: `R&D Spend`, `Administration`, `Marketing Spend`, and `State`
- Target: `Profit`

Download the dataset from [Kaggle – 50 Startups](https://www.kaggle.com/datasets/farhanmd29/50-startups) and place it at `Day2/50_Startups.csv`. The dataset is not included in this repository.

## Setup and run

From the repository root:

```bash
pip install -r requirements.txt
python Day2/multiple_linear_regression.py
```

The script uses a one-third test split with `random_state=1`, prints model metrics and coefficients, and displays actual-versus-predicted and residual plots.

## Recorded results

The previous project notes recorded the following values:

| Metric | Recorded value |
| --- | ---: |
| Test MSE | 783,376,878 |
| Test R² | 0.950 |
| Adjusted R² | 0.945 |

These values are historical notes, not a verified result from the current script revision. Re-run the script with the downloaded dataset to reproduce and confirm the metrics.

## Limitations

- This is a small, single-dataset exercise; results may not generalize to other startups or business contexts.
- Coefficients describe the fitted linear model and should not be interpreted as causal effects.
- Residual plots provide a basic diagnostic only; they do not establish that every regression assumption is satisfied.
- No cross-validation, regularization, feature scaling, or external validation is included.
