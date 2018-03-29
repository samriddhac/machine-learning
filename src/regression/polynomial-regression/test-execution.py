import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:, 1:2].values
y = dataset.iloc[:,2].values

lin_reg = LinearRegression()
lin_reg.fit(x,y)

poly_reg = PolynomialFeatures(degree = 4)
x_poly = poly_reg.fit_transform(x)
lin_poly_reg = LinearRegression()
lin_poly_reg.fit(x_poly, y)

plt.scatter(x, y, color='red')
plt.plot(x, lin_reg.predict(x), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x, y, color='red')
plt.plot(x_grid, lin_poly_reg.predict(poly_reg.fit_transform(x_grid)), color='blue')
plt.title("Truth or Bluff (Polynomial Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()


