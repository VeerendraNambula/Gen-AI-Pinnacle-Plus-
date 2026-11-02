# Applied Machine Learning & Generative AI Projects

A hands-on learning repository for implementing machine-learning concepts and documenting experiments with code, results, and visualizations.

The repository currently contains two documented projects. New work should be added as it is completed, with its own explanation, reproducible code, and results.

## Project index

| Project | Topic | Main tools |
| --- | --- | --- |
| [Day 1 – Simple Linear Regression](Day1/README.md) | Single-feature regression, model fitting, MSE and R² evaluation | Python, Pandas, NumPy, scikit-learn, Matplotlib |
| [Day 2 – Multiple Linear Regression](Day2/README.md) | Multiple predictors, categorical encoding, coefficient interpretation and regression diagnostics | Python, Pandas, NumPy, scikit-learn, Statsmodels |

## Repository structure

```text
Gen-AI-Pinnacle-Plus-/
├── Day1/
│   ├── README.md
│   ├── linear_regression.py
│   └── visualizations
├── Day2/
│   ├── README.md
│   └── multiple_linear_regression.py
└── README.md
```

## Getting started

Use Python 3.10 or later. Create a virtual environment and install the shared dependencies from `requirements.txt`.

For the Day 1 regression exercise:

```bash
pip install -r requirements.txt
python Day1/linear_regression.py
```

For the Day 2 exercise:

```bash
pip install -r requirements.txt
python Day2/multiple_linear_regression.py
```

See each project's README for its dataset, methodology, evaluation metrics, and observations.

## Topics explored

- Supervised learning and regression
- Feature preparation and categorical encoding
- Model evaluation with MSE and R²
- Statistical interpretation of regression models
- Visualization and communicating analytical results

## Adding a new project

For each new project, include:
- A short problem statement and objective.
- Dataset source and data-preparation steps.
- Reproducible setup and run instructions.
- Evaluation metrics and actual results.
- Visualizations or sample outputs where useful.
