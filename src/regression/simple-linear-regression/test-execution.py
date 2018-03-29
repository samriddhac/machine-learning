import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('Salary_Data.csv')
print(dataset)

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/3, random_state = 0)

regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

print('Real Sal ::', y_test)
print('Predicted ::', y_pred)

plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (Training Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regressor.predict(x_train), color = 'blue')
plt.title('Salary Vs Experience (Test Data)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

