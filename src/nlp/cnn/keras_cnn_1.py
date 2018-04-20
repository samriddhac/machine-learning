# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 10:53:38 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Flatten
from keras.callbacks import ModelCheckpoint

def visualize_input(img, ax):
    ax.imshow(img, cmap="gray")
    width, height = img.shape
    thresh = img.max()/2.5
    for x in range(width):
        for y in range(height):
            ax.annotate(str(round(img[x][y],2)), xy=(y,x), 
                    horizontalalignment='center',
                    verticalalignment='center',
                    color='white' if img[x][y]<thresh else 'black')
            
            

def plot_data(X_train, y_train):
    fig = plt.figure(figsize=(20,20))
    for i in range(10):
        ax = fig.add_subplot(1, 10, i+1, xticks=[], yticks=[])
        ax.imshow(X_train[i], cmap="gray")
        ax.set_title(str(y_train[i]))
    
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    visualize_input(X_train[0], ax)


def test_run():
    (X_train, y_train),(X_test, y_test) = mnist.load_data()
    print('X_train length ',len(X_train))
    print('X_test length ',len(X_test))
    
    plot_data(X_train, y_train)
    
    """re-scale the image vector """
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    
    """one hot encode the labels """
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    
    model = Sequential()
    model.add(Flatten(input_shape=X_train.shape[1:]))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    
    checkpointer = ModelCheckpoint(filepath='best.hdf5', verbose=1, save_best_only=True)

    history = model.fit(X_train, y_train, batch_size=128, epochs=10, validation_split=0.25, 
                        callbacks=[checkpointer], verbose=1, shuffle=True)    
    model.load_weights('best.hdf5')
    score = model.evaluate(X_test, y_test)
    print('Score ', score[1]*100,'%')


if __name__ == "__main__":
    test_run()  