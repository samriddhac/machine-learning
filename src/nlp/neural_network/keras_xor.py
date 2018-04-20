# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 13:13:22 2018

@author: Samriddha.Chatterjee
"""
import os
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import np_utils
from keras.utils.vis_utils import plot_model
import tensorflow as tf

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
tf.python_io.control_flow_ops = tf

np.random.seed(42)

X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y = np.array([[0],[0],[0],[1]], dtype=np.float32)

#y = np_utils.to_categorical(y)

xor = Sequential()

layer_1 = Dense(8, input_dim=X.shape[1])
xor.add(layer_1)

act_1 = Activation('tanh')
xor.add(act_1)

layer_2 = Dense(1)
xor.add(layer_2)

act_2 = Activation('sigmoid')
xor.add(act_2)

print('layer_1 ',layer_1.get_config())

xor.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
xor.summary()
history = xor.fit(x=X, y=y, epochs=1000, verbose=0)
print('history ',history)

score=xor.evaluate(X,y)
print('Accuracy ',score[-1])
print("\nPredictions:")
print(xor.predict_proba(X))

plot_model(xor, show_shapes=True, to_file='model.png')