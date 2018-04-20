# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 15:38:42 2018

@author: Samriddha.Chatterjee
"""
import numpy as np

def sigmoid(x):
    exp_x = np.exp(-x)
    return 1/(1+exp_x)
    
    
    
print(sigmoid(2))