# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 18:33:59 2018

@author: Samriddha.Chatterjee
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import SGD
from keras.utils import np_utils, to_categorical

def is_selected(admit, i):
    if admit[i] == 1:
        return True
    else:
        return False

def plot_data(dataset):
    dataset_list = dataset.values.tolist()
    y_gre = dataset.values[:,1].tolist()
    x_gpa = dataset.values[:,2].tolist()
    
    admit = dataset.values[:,0]
    
    y_gre_selected = [r for i,r in enumerate(y_gre) if is_selected(admit, i) == True]
    x_gpa_selected = [r for i,r in enumerate(x_gpa) if is_selected(admit, i) == True]
    
    y_gre_rejected = [r for i,r in enumerate(y_gre) if is_selected(admit, i) == False]
    x_gpa_rejected = [r for i,r in enumerate(x_gpa) if is_selected(admit, i) == False]
    
    plt.scatter(x_gpa_selected, y_gre_selected, color='blue')
    plt.scatter(x_gpa_rejected, y_gre_rejected, color='red')
    plt.title("GRE vs GPA")
    plt.show()
    
    for i in range(4):
        rank = i+1
        y_gre_by_rank = [r[1] for i,r in enumerate(dataset_list) if dataset_list[i][3] == rank]
        x_gpa_by_rank = [r[2] for i,r in enumerate(dataset_list) if dataset_list[i][3] == rank]
        admit_by_rank = [r[0] for i,r in enumerate(dataset_list) if dataset_list[i][3] == rank]
        
        y_gre_selected_ranked = [r for i,r in enumerate(y_gre_by_rank) if is_selected(admit_by_rank, i) == True]
        x_gpa_selected_ranked = [r for i,r in enumerate(x_gpa_by_rank) if is_selected(admit_by_rank, i) == True]
        
        y_gre_rejected_ranked = [r for i,r in enumerate(y_gre_by_rank) if is_selected(admit_by_rank, i) == False]
        x_gpa_rejected_ranked = [r for i,r in enumerate(x_gpa_by_rank) if is_selected(admit_by_rank, i) == False]
        
        plt.scatter(x_gpa_selected_ranked, y_gre_selected_ranked, color='blue')
        plt.scatter(x_gpa_rejected_ranked, y_gre_rejected_ranked, color='red')
        plt.title("GRE vs GPA by Rank "+str(rank))
        plt.show()    
    

def train_test_data(data):
    """Remove na with 0"""
    data = data.fillna(0)
    """One hot encoding the rank"""
    processed_data = pd.get_dummies(data, columns=["rank"])
    """Normalizing the values , place all data within range on (0,1)"""
    processed_data["gre"] = processed_data["gre"]/800
    processed_data["gpa"] = processed_data["gpa"]/4
    
    """split data"""
    X=np.array(processed_data)[:,1:].astype('float32')
    y = to_categorical(data["admit"],num_classes=2)
    
    print('Shape of X ', X.shape)
    print('Shape of y ', y.shape)
    
    """ Split train/test data """
    X_train, X_test = X[50:], X[:50]
    y_train, y_test = y[50:], y[:50]
    
    model=Sequential()
    model.add(Dense(128, activation='relu', input_dim=X.shape[1]))
    model.add(Dropout(0.2))
    model.add(Dense(64,activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation='softmax'))
              
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    
    model.summary()
    
    model.fit(X_train, y_train, epochs=200, batch_size=100, verbose=0)
    
    score = model.evaluate(X_train, y_train)
    print('Training score ', score[1])
    score = model.evaluate(X_test, y_test)
    print('Testing score ', score[1])
    
    

dataset = pd.read_csv('https://stats.idre.ucla.edu/stat/data/binary.csv')
plot_data(dataset)
train_test_data(dataset)




    
