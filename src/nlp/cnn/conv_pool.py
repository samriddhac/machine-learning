# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 10:35:53 2018

@author: Samriddha.Chatterjee
"""

from keras.models import Sequential
from keras.layers.convolutional import MaxPooling2D

model = Sequential()
model.add(MaxPooling2D(pool_size=8, strides=8, padding='same', input_shape=(200,200,15)))

model.summary()