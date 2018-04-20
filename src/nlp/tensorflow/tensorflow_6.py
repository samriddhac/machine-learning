# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 13:54:22 2018

@author: Samriddha.Chatterjee
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

model_file = './trained_model.ckpt'

mnist = input_data.read_data_sets('.', one_hot=True, reshape=False)

test_features = mnist.test.images
test_labels = mnist.test.labels.astype('float32')

learning_rate = 0.001
epochs = 20
batch_size = 128

n_input =784
n_classes = 10

n_hidden_layer_1 = 512 #No of feature map
n_hidden_layer_2 = 256

x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, n_classes])

x_flat =tf.reshape(x, [-1, n_input])

weights = {
    'hidden_layer_1': tf.Variable(tf.random_normal([n_input, n_hidden_layer_1])),
    'hidden_layer_2': tf.Variable(tf.random_normal([n_hidden_layer_1, n_hidden_layer_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_layer_2, n_classes]))
}

biases = {
    'hidden_layer_1': tf.Variable(tf.random_normal([n_hidden_layer_1])),
    'hidden_layer_2': tf.Variable(tf.random_normal([n_hidden_layer_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

keep_prob = tf.placeholder(tf.float32)

layer_1 = tf.add(tf.matmul(x_flat, weights['hidden_layer_1']), biases['hidden_layer_1'])
layer_1 = tf.nn.relu(layer_1)
layer_1 = tf.nn.dropout(layer_1, keep_prob)
layer_2 = tf.add(tf.matmul(layer_1, weights['hidden_layer_2']), biases['hidden_layer_2'])
layer_2 = tf.nn.relu(layer_2)
layer_2 = tf.nn.dropout(layer_2, keep_prob)
logits = tf.add(tf.matmul(layer_2, weights['out']), biases['out'])

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

predictions = tf.nn.softmax(logits)

correct_prediction = tf.equal(tf.argmax(predictions, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as session:
    session.run(init)
    for i in range(epochs):
        total_batch = int(mnist.train.num_examples/batch_size)
        for j in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            session.run(optimizer, feed_dict={x:batch_x, y:batch_y, keep_prob:0.8})
        
        validation_accuracy = session.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, keep_prob:1.0})
        print('Epoch {:<3} - validation accuracy {}'.format(i, validation_accuracy))
    
    saver.save(session, model_file)
    print('Trained Model Saved.')
    
    saver.restore(session, model_file)
    score = session.run(accuracy, feed_dict={x:test_features, y:test_labels, keep_prob:1.0}) 
    print('Test Score ',score)
    
    
