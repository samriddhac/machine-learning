# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 13:26:15 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.transform import resize 
from keras.models import Sequential
from keras.layers.convolutional import Conv2D


def apply_filter(model, img, i, filters, ax):
    print(filters[i].shape)
    print(np.reshape(filters[i], (4,4,1,1)))
    #Converts into 4-D matrix
    model.layers[0].set_weights([np.reshape(filters[i], (4,4,1,1)), np.array([0])])
    # plot the corresponding activation map
    ax.imshow(np.squeeze(model.predict(np.reshape(img, (1, img.shape[0], img.shape[1], 1)))), cmap='gray')

def test_run():
    img_path = 'images/udacity_sdc.png'
    bgr_img = cv2.imread(img_path)
    gray_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    small_img = resize(gray_img, (round(gray_img.shape[0]*3/10), round(gray_img.shape[1]*3/10)), mode='constant')
    small_img = small_img.astype('float32')/255
    plt.imshow(small_img, cmap='gray')
    plt.show()

    filter_vals = np.array([[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1],[1,1,-1,-1]])    
    filter_1 = filter_vals
    filter_2 = -filter_1
    filter_3 = filter_1.T
    filter_4 = -filter_3
    filters = [filter_1, filter_2, filter_3, filter_4]
    
    fig = plt.figure(figsize=(10,5))
    for i in range(4):
        ax = fig.add_subplot(1,4,i+1, xticks=[], yticks=[])
        ax.imshow(filters[i], cmap="gray")
        width, height = filters[i].shape
        for x in range(width):
            for y in range(height):
                ax.annotate(str(filters[i][x][y]), xy=(x,y),
                            color= 'white' if filters[i][x][y]<0 else 'black',
                            horizontalalignment='center',
                            verticalalignment='center')
    
    model = Sequential()
    model.add(Conv2D(filters=1, kernel_size=4, strides=1, padding='same',
                     activation='relu', input_shape=(small_img.shape[0], small_img.shape[1], 1)))
    fig = plt.figure(figsize=(20,20))
    for i in range(4):
        ax = fig.add_subplot(1,4,i+1, xticks=[], yticks=[])
        apply_filter(model, small_img, i, filters, ax)
        ax.set_title('Activation Map for Filter %s' % str(i+1))
    
    #model.summary()
    
if __name__=='__main__':
    test_run()