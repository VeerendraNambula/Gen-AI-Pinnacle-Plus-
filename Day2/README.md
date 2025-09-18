# Day 2: Develop a Multiple Regression Model

> Predicting outcomes using multiple independent variables with full statistical evaluation.

This project implements **Multiple Linear Regression** to model the relationship between several predictors and a continuous response variable. We evaluate significance, fit, and assumptions using both machine learning and statistical approaches.

---

## 🎯 Objectives
- Implement multiple regression using `scikit-learn` and `statsmodels`
- Handle categorical variables via one-hot encoding
- Interpret regression coefficients and p-values
- Evaluate model performance using:
  - MSE
  - R² and Adjusted R²
  - Residual plots
- Perform basic feature selection

---

## 🧪 Dataset
Used: **50_Startups.csv**  
Features: `R&D Spend`, `Administration`, `Marketing Spend`, `State`  
Target: `Profit`

👉 Download: [Kaggle - 50_Startups](https://www.kaggle.com/farhanmd29/50-startups)

If not available, synthetic data is generated in code.

---

## 📊 Results

| Metric | Value |
|-------|-------|
| Mean Squared Error (MSE) | 783,376,878 |
| R² Score | 0.950 |
| Adjusted R² | 0.945 |

✅ Strong explanatory power; R&D Spend is the most significant predictor.

---

## 📈 Key Insights
- **R&D Spend** has the highest positive impact on profit.
- Marketing Spend shows moderate influence.
- Administration cost is less significant.
- Assumptions checked: Linearity, Homoscedasticity, Normality of residuals.

---

## 🛠 Requirements

Install dependencies:
```bash
pip install numpy pandas scikit-learn statsmodels matplotlib seaborn
