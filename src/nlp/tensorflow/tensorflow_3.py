# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 14:30:23 2018

@author: Samriddha.Chatterjee
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np

def batches(features, labels, batch_size):
    n_samples = len(features)
    assert len(features) == len(labels), 'Feature and Labels length should be same'
    out_batch =[]
    for start_i in range(0, n_samples, batch_size):
        end_i = start_i+batch_size
        batch = [features[start_i:end_i], labels[start_i:end_i]]
        out_batch.append(batch)
        
    return out_batch

def print_epoch_stats(i_epoch, last_features, last_labels, session):
    current_cost = session.run(cost, 
                               feed_dict={features:last_features, labels:last_labels})
    valid_accuracy = session.run(accuracy, 
                                 feed_dict={features:valid_features, labels:valid_labels})
    print('Epoch: {:<4} - Cost: {:<8.3} Validation Accuracy: {:<5.3}'.format(
        i_epoch,
        current_cost,
        valid_accuracy))

n_input = 784
n_classes =10
learning_rate = 0.1

mnist = input_data.read_data_sets('/datasets/ud730/mnist', one_hot=True)

train_features = mnist.train.images
test_features = mnist.test.images
valid_features = mnist.validation.images

train_labels = mnist.train.labels.astype('float32')
test_labels = mnist.test.labels.astype('float32')
valid_labels = mnist.validation.labels.astype('float32')

features = tf.placeholder(tf.float32, [None, n_input])
labels = tf.placeholder(tf.float32, [None, n_classes])

weights = tf.Variable(tf.random_normal([n_input, n_classes]))
biases = tf.Variable(tf.random_normal([n_classes]))

logits = tf.add(tf.matmul(features, weights), biases)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

batch_size = 128
epochs = 100
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    for i in range(epochs):
        for batch_features, batch_labels in batches(train_features, train_labels, batch_size):
            session.run(optimizer, feed_dict={ features: batch_features, labels:batch_labels})
        print_epoch_stats(i, batch_features, batch_labels, session)
     
    test_accuracy = session.run(accuracy, feed_dict={features:test_features, labels:test_labels})
    print('Test Accuracy: {}'.format(test_accuracy))    
    





