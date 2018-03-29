import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv('sam_salary.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values
print(x)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/4, random_state = 0)

plt.scatter(x, y, color = 'green')
plt.title('Salary Plot')
plt.xlabel('Experience')
plt.ylabel('Salary(By year)')
plt.show()

regressor = LinearRegression()
regressor.fit(x_train, y_train)
y_predict = regressor.predict(x_test)

print('Actual Salary ',y_test)
print('Predicted Salary ',y_predict)

plt.scatter(x_train, y_train, color = 'blue')
plt.plot(x_train, regressor.predict(x_train), color = 'red')
plt.title('Salary Plot (Training Data)')
plt.xlabel('Experience')
plt.ylabel('Salary(By year)')
plt.show()

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
ploy_reg_model = LinearRegression()
ploy_reg_model.fit(x_poly, y)

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color ='red')
plt.plot(x, ploy_reg_model.predict(poly_reg.fit_transform(x)), color = 'blue')
plt.title('Salary Vs Experience (Polynomial Regression)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

test_predict = [ [12.0], [13.0], [14.0], [15.0]]
sal_predict = regressor.predict(test_predict)
print('Linear predictions ',sal_predict)
sal_predict_2 = ploy_reg_model.predict(poly_reg.fit_transform(test_predict))
print('Polynomial predictions ',sal_predict_2)