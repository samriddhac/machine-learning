# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:52:53 2018

@author: Samriddha.Chatterjee
"""

import numpy as np

def softmax(L):
    expL = np.exp(L)
    print(expL)
    sumExpL = sum(expL)
    print(sumExpL)
    result = []
    for i in expL:
        result.append(i*1.0/sumExpL)
    return result

print(softmax([9,4,6,8]))