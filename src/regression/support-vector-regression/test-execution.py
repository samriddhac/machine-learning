import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

dataset = pd.read_csv('Position_Salaries.csv')
x = dataset.iloc[:,1:2].values
y = dataset.iloc[:,2].values

sc_x = StandardScaler()
sc_y = StandardScaler()
x = sc_x.fit_transform(x)
y = sc_y.fit_transform(y.reshape((len(y), 1)))

regressor = SVR(kernel='rbf')
regressor.fit(x,y)

plt.scatter(x, y, color='red')
plt.plot(x, regressor.predict(x), color='blue')
plt.title("Truth or Bluff (Linear Regression)")
plt.xlabel("Levels")
plt.ylabel("Salary")
plt.show()

X_grid = np.arange(min(x), max(x), 0.01) # choice of 0.01 instead of 0.1 step because the data is feature scaled
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(x, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(sc_y.inverse_transform(regressor.predict(sc_x.transform(np.array([[6.5]])))))