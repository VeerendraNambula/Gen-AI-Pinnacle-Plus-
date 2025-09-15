# Day 1: Develop a Simple Linear Regression Model

> Predictive Analysis with Evaluation Metrics (MSE, R²)

Build a simple linear regression model to predict a target variable based on a single feature. Evaluate performance using standard metrics.

---

## 🎯 Objective
- Implement linear regression from scratch and using **scikit-learn**
- Visualize the best-fit line
- Evaluate model using **MSE** and **R² score**

---

## 🧪 Dataset
Used sample dataset: `Salary_Data.csv` (YearsExperience vs Salary)  
Download link: [Kaggle - Salary Dataset](https://www.kaggle.com/datasets/karthickveerakumar/salary-data-simple-linear-regression)

---

## 📊 Results

| Metric    | Training Set | Test Set |
|-----------|--------------|----------|
| MSE       | 27,228,701.91 | 50,629,462.73 |
| RMSE      | 5,218.11 | 7,115.44 |
| R² Score  | 0.9644 | 0.9009 |

✅ High accuracy indicating good fit.

**Predictions:**
- Salary for **9.3 years** of experience: \$113,240.73  
- Salary for **5 years** of experience: \$72,251.34  

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
