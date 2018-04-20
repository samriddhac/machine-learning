# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 16:53:33 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical

np.random.seed(42)

def test_run():
    (x_train, y_train),(x_test, y_test) = imdb.load_data(num_words=1000)
    print('X shape ', x_train.shape)
    print('Y shape ', y_train.shape)
    
    tokenizer = Tokenizer(num_words=1000)
    x_train = tokenizer.sequences_to_matrix(x_train, mode='binary')
    x_test = tokenizer.sequences_to_matrix(x_test, mode='binary')
    
    print(x_train)
    
    num_classes=2
    y_train = to_categorical(y_train, num_classes=num_classes)
    y_test = to_categorical(y_test, num_classes=num_classes)
    
    model = Sequential()
    model.add(Dense(512, activation='tanh', input_dim=x_train.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    model.summary()
    
    history = model.fit(x_train, y_train, epochs=10, batch_size=32, 
              validation_data=(x_test, y_test), verbose=1)
    
    score = model.evaluate(x_test, y_test)
    print('Accuracy ', score[0])
    
    
if __name__ == "__main__":
    test_run()    
