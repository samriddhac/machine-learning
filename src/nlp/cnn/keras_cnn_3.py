# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 13:20:52 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Flatten
from keras.utils.np_utils import to_categorical
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

def plot_data(X_train, y_train):
    fig = plt.figure(figsize=(20,8))
    for i in range(40):
        ax = fig.add_subplot(5,8,i+1,xticks=[], yticks=[])
        ax.imshow(X_train[i])
        ax.set_title(y_train[i])

def test_run():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    plot_data(X_train, y_train)
    
    #Scale the data
    X_train = X_train.astype('float32')/255
    X_test = X_test.astype('float32')/255
    
    num_classes = len(np.unique(y_train))
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)
    
    X_train, X_valid = X_train[5000:], X_train[:5000]
    y_train, y_valid = y_train[5000:], y_train[:5000]
    
    
    model = Sequential()
    
    model.add(Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(32,32,3)))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=32, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Conv2D(filters=64, kernel_size=2, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2, strides=2))
    model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(500, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    
    model.summary()
    
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    checkpoint = ModelCheckpoint(filepath='best_cnn_3.hdf5', save_best_only=True)
    
    history = model.fit(X_train, y_train, batch_size=32, epochs=100, verbose=2,
                        validation_data=(X_valid, y_valid),
                        callbacks=[checkpoint], shuffle=True)
    
    model.load_weights('best_cnn_3.hdf5')
    score = model.evaluate(X_test, y_test)
    
    print('Accuracy {}'.format(score[1]))
    
    y_hat = model.predict(X_test[:20])
    print(y_hat)
    cifar10_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    fig = plt.figure(figsize=(20, 8))
    for idx in range(20):
        ax = fig.add_subplot(4, 8, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(X_test[idx]))
        pred_idx = np.argmax(y_hat[idx])
        true_idx = np.argmax(y_test[idx])
        ax.set_title("{} ({})".format(cifar10_labels[pred_idx], cifar10_labels[true_idx]),
                     color=("green" if pred_idx == true_idx else "red"))
    
if __name__=='__main__':
    test_run()    