# -*- coding: utf-8 -*-
"""
Created on Wed Apr 18 17:46:15 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
import matplotlib.pyplot as plt

def y_val(s_1, w, x):
    y =[]
    for idx, i in enumerate(x):
        if idx == 0:
            y.append(s_1)
        else: 
            s_1 = w*s_1 - w* np.square(s_1)
            y.append(s_1)
    return y

def plot_data(w, s_1):
    x_plot = np.linspace(0, 50, 51, dtype='float32')
    y_plot = y_val(s_1, w, x_plot)
    
    plt.scatter(x_plot, y_plot, color='blue')
    plt.plot(x_plot, y_plot, color='red')
    plt.show()
    
if __name__=='__main__':
    plot_data(1,0.5)
    plot_data(3,0.5)
    plot_data(4,0.0001)