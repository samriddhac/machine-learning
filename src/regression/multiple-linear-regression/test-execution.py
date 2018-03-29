import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as sm

dataset = pd.read_csv('50_Startups.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

labelEncoder = LabelEncoder()
x[:,3] = labelEncoder.fit_transform(x[:,3])
oneHotEncoder = OneHotEncoder(categorical_features = [3])
x = oneHotEncoder.fit_transform(x).toarray()

#Avoid dummy variable trap
x = x[:,1:]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1/5, random_state = 0)


reggressor = LinearRegression()
reggressor.fit(x_train, y_train)

y_pred = reggressor.predict(x_test)

N = len(y_pred)
index = np.arange(N)
width = 0.35

fig, ax = plt.subplots()
actuals_rects = ax.bar(index, y_test, width, color = 'green')
predict_rects = ax.bar(index+width, y_pred, width, color = 'blue')
ax.set_ylabel('Profits')
ax.set_title('Actual vs Predictions')
ax.set_xticks(index + width / 2)
ax.set_xticklabels(index)

ax.legend((actuals_rects[0], predict_rects[0]), ('Actual', 'Predictions'))

plt.show()

x = np.append(arr = np.ones((50,1)).astype(int), values = x, axis = 1)

x_opt = x[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
x_opt = x[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
x_opt = x[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
x_opt = x[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
x_opt = x[:, [0, 3]]
regressor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
print(regressor_OLS.summary())
print(x_opt)

x_train, x_test, y_train, y_test = train_test_split(x_opt, y, test_size = 1/5, random_state = 0)
reggressor = LinearRegression()
reggressor.fit(x_train, y_train)

y_pred = reggressor.predict(x_test)

N = len(y_pred)
index = np.arange(N)
width = 0.35

fig, ax = plt.subplots()
actuals_rects = ax.bar(index, y_test, width, color = 'green')
predict_rects = ax.bar(index+width, y_pred, width, color = 'blue')
ax.set_ylabel('Profits')
ax.set_title('Actual vs Predictions(Backward Elimination)')
ax.set_xticks(index + width / 2)
ax.set_xticklabels(index)

ax.legend((actuals_rects[0], predict_rects[0]), ('Actual', 'Predictions'))

plt.show()