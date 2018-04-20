# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 12:59:11 2018

@author: Samriddha.Chatterjee
"""

import tensorflow as tf

output = None
hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

weights = [
        tf.Variable(hidden_layer_weights),
        tf.Variable(out_weights)
        ]

biases = [
        tf.Variable(tf.zeros(3)),
        tf.Variable(tf.zeros(2))
        ]

features = tf.Variable([[1.0, 2.0, 3.0, 4.0], [-1.0, -2.0, -3.0, -4.0], [11.0, 12.0, 13.0, 14.0]])
print(features.shape)
hidden_layer = tf.add(tf.matmul(features, weights[0]), biases[0])
print(hidden_layer.shape)
hidden_layer = tf.nn.relu(hidden_layer)
print(hidden_layer.shape)
logits = tf.add(tf.matmul(hidden_layer, weights[1]), biases[1])

init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    out = session.run(logits)
    print(out)