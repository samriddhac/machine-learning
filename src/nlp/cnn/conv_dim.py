# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 21:47:17 2018

@author: Samriddha.Chatterjee
"""

from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))
model.summary()