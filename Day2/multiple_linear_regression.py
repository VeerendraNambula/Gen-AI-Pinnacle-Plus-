import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv(r"C:\Users\nambu\Downloads\50_Startups.csv")
dataset.head()

X = dataset.drop(['Profit'],axis=1)
y = dataset[['Profit']]

X = pd.get_dummies(X,drop_first=True)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=1)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Training Dataset
pred_train = regressor.predict(X_train)

# Testing Dataset
pred_test = regressor.predict(X_test)

plt.plot(np.array(y_train))
plt.plot(pred_train)
plt.title('Training Data- Actual vs Predicted')
plt.legend(['Actual','Predicted'])

plt.plot(np.array(y_test))
plt.plot(pred_test)
plt.title('Testing Data- Actual vs Predicted')
plt.legend(['Actual','Predicted'])

plt.scatter(y_test, pred_test)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],'r--')
plt.title('Acutal vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')