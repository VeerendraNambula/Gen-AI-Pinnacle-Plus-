# Day 1: Simple Linear Regression from Scratch

Implement a single-feature linear regression model with batch gradient descent and evaluate its predictions on a held-out test set.

## Learning objectives

- Implement gradient descent without using a regression estimator.
- Standardize the feature and target during optimization for stable convergence.
- Convert predictions back to the original salary scale.
- Evaluate with MAE, RMSE, and R².
- Visualize the fitted line and optimization cost.

## Dataset

The script expects `Salary_Data.csv` in this directory with columns:

- `YearsExperience`
- `Salary`

Download the dataset from [Kaggle – Salary Dataset](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression). The dataset is not committed to this repository.

## Setup and run

From the repository root:

```bash
pip install -r requirements.txt
python Day1/linear_regression.py
```

The script uses an 80/20 train/test split with `random_state=42`, prints train and test metrics, predicts salary for five years of experience, and displays the fitted line and cost history.

## Results

Metrics are calculated from the downloaded dataset at runtime. No fixed benchmark is published because the dataset version and current implementation should be evaluated together.

## Implementation notes

The gradient-descent model standardizes its input feature and target internally. This avoids unstable updates caused by optimizing directly against salary values with a relatively large numeric scale. Predictions are returned in the original salary units.

The implementation is intended for learning and is not a substitute for a validated compensation model.
