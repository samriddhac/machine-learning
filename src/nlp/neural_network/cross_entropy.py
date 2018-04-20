# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 12:26:54 2018

@author: Samriddha.Chatterjee
"""
import numpy as np

def cross_entropy(Y, p):
    Y = np.float_(Y)
    p = np.float_(p)
    return -np.sum(Y*np.log(p) + (1-Y)*np.log(1-p))