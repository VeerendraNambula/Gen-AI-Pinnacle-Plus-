# Day 1: Develop a Simple Linear Regression Model

> Predictive Analysis with Evaluation Metrics (MSE, R²)

Build a simple linear regression model to predict a target variable based on a single feature. Evaluate performance using standard metrics.

---

## 🎯 Objective
- Implement linear regression from scratch and using scikit-learn
- Visualize the best-fit line
- Evaluate model using MSE and R² score

---

## 🧪 Dataset
Used sample dataset: `Salary_Data.csv` (YearsExperience vs Salary)  
Download link: [Kaggle - Salary Dataset](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)
---

## 📊 Results
| Metric | Value |
MSE on Training Set: 27228701.908034485
RMSE on Training Set: 5218.112868464468
R2 Score on Training Set: 0.9643747107705292

MSE on Test Set: 50629462.73329226
RMSE on Test Set: 7115.4383373965275
R2 Score on Test Set: 0.9008812357081055

✅ High accuracy indicating good fit.

Predicted salary for 9.3 years of experience will be $113240.73475129377
Predicted salary for 5 years of experience will be $72251.34092530378

---

## 🛠 Tech Stack
- Python
- Scikit-learn
- Matplotlib
- Pandas
- NumPy

---

## 💻 Code Usage

Install dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib
