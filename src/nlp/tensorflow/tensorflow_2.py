# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:59:43 2018

@author: Samriddha.Chatterjee
"""

import tensorflow as tf


def get_weights(n_features, n_labels):
    return tf.Variable(tf.truncated_normal((n_features, n_labels)))

def get_bias(n_labels):
    return tf.Variable(tf.zeros(n_labels))

def linear(input, w, b):
    return tf.Variable(tf.add(tf.matmul(input, w), b))

logit_data = [2.0, 1.0, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

logits = tf.placeholder(tf.float32)
softmax = tf.nn.softmax(logits)
one_hot = tf.placeholder(tf.float32)

cross_entropy = -tf.reduce_sum(tf.multiply(one_hot, tf.log(softmax)))

x = tf.Variable(5)
n_features = 120
n_labels = 5
weights = tf.Variable(tf.truncated_normal((n_features, n_labels)))
bias = tf.Variable(tf.zeros(n_labels))
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    session.run(x)
    o_weights = session.run(weights)
    print(o_weights)
    softmax_output = session.run(softmax, feed_dict={logits:logit_data})
    print(softmax_output)
    cross_entropy_output = session.run(cross_entropy, feed_dict={one_hot:one_hot_data, softmax:softmax_output})
    print(cross_entropy_output)
    
    