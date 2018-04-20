# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 13:10:08 2018

@author: Samriddha.Chatterjee
"""
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.callbacks import ModelCheckpoint

x_train = np.array([1,3,5,7,9,11,13,15,17,19,21])
y_train = np.array([3,5,7,9,11,13,15,17,19,21,23])


x_test = np.array([23,25,27,29,31,33,35,37,39])
y_test = np.array([25,27,29,31,33,35,37,39,41])

x_test_1 = np.array([23,25])
x_test_1 = np.reshape(x_test_1,(2, 1))
print(x_test_1)

model = Sequential()
model.add(Dense(1, input_dim=1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint(filepath='best.m.checkpoint.hdf5', save_best_only=True)

history = model.fit(x_train, y_train, epochs=3000, batch_size=3, callbacks=[checkpoint], 
                    verbose=0,
                    validation_split=0.2)
score = model.evaluate(x_test, y_test)
print('Score ', score[1])

print('Weights ', model.get_weights())

w_1 = model.get_weights()[0][0][0]
w_0 = model.get_weights()[1][0]

print('w_0 :', w_0)
print('w_1 :', w_1)

x_input = np.array([41, 43, 45, 47]).astype('float32')
out = w_0 + w_1 * x_input
print('Generated outputs ', out)


