import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

#Import dataset
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,3].values
print(dataset)
print(x)
print(y)

#Handle missing data
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis = 0)
imputer = imputer.fit(x[:,1:3])
x[:,1:3] = imputer.transform(x[:,1:3])
print(x)

#Encode Categorical variables .
labelEncoder_x =  LabelEncoder()
x[:, 0] = labelEncoder_x.fit_transform(x[:,0])
oneHotEncoder = OneHotEncoder(categorical_features=[0])
x = oneHotEncoder.fit_transform(x).toarray()
print(x)

labelEncoder_y =  LabelEncoder()
y = labelEncoder_y.fit_transform(y)
print(y)


#Train & test data split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

#Feature scaling
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)