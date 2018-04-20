# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# Setting the random seed, feel free to change it and see different solutions.
np.random.seed(42)

def stepFunction(t):
    if t >= 0:
        return 1
    return 0

def prediction(X, W, b):
    return stepFunction((np.matmul(X,W)+b)[0])

# TODO: Fill in the code below to implement the perceptron trick.
# The function should receive as inputs the data X, the labels y,
# the weights W (as an array), and the bias b,
# update the weights and bias W, b, according to the perceptron algorithm,
# and return W and b.
def perceptronStep(X, y, W, b, learn_rate = 0.01):
    X = X.values
    y = y.values
    for i in range(len(X)):
        y_hat = prediction(X[i], W, b)
        if y[i]-y_hat == 1:
            W[0] += X[i][0]*learn_rate
            W[1] += X[i][1]*learn_rate
            b += learn_rate
        elif y[i]-y_hat == -1:
            W[0] -= X[i][0]*learn_rate
            W[1] -= X[i][1]*learn_rate
            b -= learn_rate
    return W, b
    
# This function runs the perceptron algorithm repeatedly on the dataset,
# and returns a few of the boundary lines obtained in the iterations,
# for plotting purposes.
# Feel free to play with the learning rate and the num_epochs,
# and see your results plotted below.
def trainPerceptronAlgorithm(X, y, learn_rate = 0.01, num_epochs = 100):
    x_min, x_max = min(X.T[0]), max(X.T[0])
    y_min, y_max = min(X.T[1]), max(X.T[1])
    W = np.array(np.random.rand(2,1))
    b = np.random.rand(1)[0] + x_max
    # These are the solution lines that get plotted below.
    boundary_lines = []
    for i in range(num_epochs):
        # In each epoch, we apply the perceptron step.
        W, b = perceptronStep(X, y, W, b, learn_rate)
        boundary_lines.append((-W[0]/W[1], -b/W[1]))
    return boundary_lines

def plot(dataset, boundary_lines):
    blue_dataset = []
    red_dataset = []
    for item in dataset:
        if item[2] == 1:
            blue_dataset.append(item[:2])
        else:
            red_dataset.append(item[:2])
    plt.scatter(np.array(blue_dataset)[:,0], np.array(blue_dataset)[:,1], color='blue')
    plt.scatter(np.array(red_dataset)[:,0], np.array(red_dataset)[:,1], color='red')
    
    num_lines = 0
    for line in boundary_lines:
        x_plots = np.linspace(0,1,5)
        y_plots = []
        for x in x_plots:
            y = line[0]*x + line[1]
            print(y)
            y_plots.append(y)
            
        mx_y = max(y_plots)
        if mx_y<2:
            if num_lines ==len(boundary_lines)-1:
                plt.plot(x_plots, y_plots, color='black')
            else:
                plt.plot(x_plots, y_plots, color='green')
        num_lines=num_lines+1
    
    plt.show()

def test_run():
    dataset = pd.read_csv('data.csv')
    X = pd.DataFrame(data=dataset.values[:, :-1])
    y = pd.DataFrame(data=dataset.values[:, 2])
    boundary_lines = trainPerceptronAlgorithm(X, y)
    plot(dataset.values, boundary_lines)

if __name__ == "__main__":
    test_run()
