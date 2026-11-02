# Day 2: Multiple Linear Regression

Estimate startup profit from research and development spend, administration spend, marketing spend, and state.

## Learning objectives

- Prepare numeric and categorical predictors.
- Fit multiple linear regression with scikit-learn.
- Evaluate predictions on a held-out test set.
- Interpret coefficients cautiously and inspect residual patterns.
- Reuse the model preparation and evaluation functions in tests or other scripts.

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

The script uses a one-third test split with `random_state=1`, prints train/test metrics and coefficients, and displays actual-versus-predicted and residual plots.

## Tests

```bash
pip install pytest
python -m pytest -q
```

The tests use synthetic data and do not require the external CSV dataset.

## Results

Metrics are calculated from the dataset at runtime. No fixed benchmark is claimed; rerun the current implementation against the downloaded dataset to obtain reproducible values.

## Limitations

- This is a small, single-dataset exercise; results may not generalize to other startups or business contexts.
- Coefficients describe the fitted linear model and should not be interpreted as causal effects.
- Residual plots provide basic diagnostics and do not establish that every regression assumption is satisfied.
- Cross-validation, regularization, and external validation are not implemented.
