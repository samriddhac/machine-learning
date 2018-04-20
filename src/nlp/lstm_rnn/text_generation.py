# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 14:33:10 2018

@author: Samriddha.Chatterjee
"""

import time
from collections import namedtuple

import numpy as np
import tensorflow as tf
import pandas as pd

#a = 'this is text'
#print(sorted(set(a)))

t1 = np.array([[[1, 2], [2, 3]], [[4, 4], [5, 3]]])
print(t1.shape)
s = tf.concat(t1, axis=1)
with tf.Session() as sess:
    out = sess.run(s)
    print(out)
    
    
def build_loss(logits, targets, lstm_size, num_classes):
    y_one_hot = tf.one_hot(targets, num_classes)
    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped)
    loss = tf.reduce_mean(loss)
    return loss

def build_optimizer(loss, learning_rate, grade_clip):
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grade_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    return optimizer

"""If our input has batch size  NN , number of steps  MM , and the hidden layer has  L hidden units, 
then the output is a 3D tensor with size  N×M×L . The output of each LSTM cell has size  L , 
we have  M  of them, one for each sequence step, and we have  N  sequences. So the total size is  N×M×L.
We are using the same fully connected layer, the same weights, for each of the outputs. 
Then, to make things easier, we should reshape the outputs into a 2D tensor with shape  (M∗N)×L . 
That is, one row for each sequence and step, where the values of each row are the output from the LSTM cells."""

def build_output(lstm_output, in_size, out_size):
    print('lstm_output ',lstm_output)
    print('in_size ',in_size)
    print('out_size ',out_size)
    seq_output = tf.concat(lstm_output, axis=1)
    x = tf.reshape(seq_output, [-1, in_size])
    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    logits = tf.add(tf.matmul(x,softmax_w), softmax_b)
    out = tf.nn.softmax(logits, name='predictions')
    return out, logits

def build_inputs(batch_size, n_steps):
    inputs = tf.placeholder(tf.int32, [batch_size, n_steps], name='inputs_0')
    targets = tf.placeholder(tf.int32, [batch_size, n_steps], name='targets_0')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob_0')
    
    return inputs, targets, keep_prob


def build_cell(lstm_size, keep_prob):
    lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
    drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop


def build_lstm(lstm_size, num_layers, batch_size, keep_prob):
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(lstm_size, keep_prob) for _ in range(num_layers)])
    initial_state = cell.zero_state(batch_size, tf.float32)
    return cell, initial_state
    

"""batch_size ==> no of sequence """

def get_batches(arr, batch_size, n_steps):
    char_per_batch = n_steps*batch_size
    print('char_per_batch ',char_per_batch)
    
    n_batch = len(arr)//char_per_batch
    print('n_batch ',n_batch)
    
    knd = int(char_per_batch*n_batch)
    arr = arr[:knd]
    print('Total char ', knd)
    
    arr = arr.reshape((batch_size, -1))
    print('arr shape ', arr.shape)
    
    for n in range(0, arr.shape[1], n_steps):
        x = arr[:, n:n+n_steps]
        y_temp = arr[:, n+1:n+n_steps+1]
        
        print('x shape ', x.shape)
        #print('y_temp shape ', y_temp.shape)
        
        y = np.zeros(x.shape, dtype=x.dtype)
        y[:, :y_temp.shape[1]] = y_temp
        yield x,y
        
        
        
class CharRNN:
    def __init__(self, num_classes, batch_size=64, num_steps=50, lstm_size=128,
                 num_layers=2, learning_rate=0.001, grad_clips=5,
                 sampling=False):
        if sampling == True:
            batch_size, num_steps = 1, 1
        else:
            batch_size, num_steps = batch_size, num_steps
            
        self.inputs, self.targets, self.keep_prob = build_inputs(batch_size, num_steps)
        
        cell, self.initial_state = build_lstm(lstm_size, num_layers, batch_size, self.keep_prob)
        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        lstm_output, states = tf.nn.dynamic_rnn(cell, x_one_hot, initial_state=self.initial_state)
        self.final_state = states
        
        self.predictions, self.logits = build_output(lstm_output, lstm_size, num_classes)
        
        self.loss = build_loss(self.logits, self.targets, lstm_size, num_classes)
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clips)
        
        

"""
batch_size - Number of sequences running through the network in one pass.
num_steps - Number of characters in the sequence the network is trained on. Larger is better typically, 
the network will learn more long range dependencies. But it takes longer to train. 
100 is typically a good number here.
lstm_size - The number of units in the hidden layers.
num_layers - Number of hidden LSTM layers to use
learning_rate - Learning rate for training
keep_prob - The dropout keep probability when training. If you're network is overfitting, try decreasing this.
"""

"""
If your training loss is much lower than validation loss then 
this means the network might be overfitting. Solutions to this are to 
decrease your network size, or to increase dropout. For example you could try 
dropout of 0.5 and so on.
If your training/validation loss are about equal then your model is underfitting. 
Increase the size of your model (either number of layers or the raw number of 
neurons per layer)
"""

"""
The two most important parameters that control the model are lstm_size and num_layers.
I would advise that you always use num_layers of either 2/3. The lstm_size can be 
adjusted based on how much data you have

The two important quantities to keep track of here are:

The number of parameters in your model. This is printed when you start training.
The size of your dataset. 1MB file is approximately 1 million characters.

I have a 100MB dataset and I'm using the default parameter settings 
(which currently print 150K parameters). My data size is significantly larger 
(100 mil >> 0.15 mil), so I expect to heavily underfit. I am thinking I can 
comfortably afford to make lstm_size larger.
I have a 10MB dataset and running a 10 million parameter model. I'm slightly 
nervous and I'm carefully monitoring my validation loss. If it's larger than 
my training loss then I may want to try to increase dropout a bit and see 
if that helps the validation loss.

The winning strategy to obtaining very good models (if you have the compute time) 
is to always err on making the network larger (as large as you're willing to wait 
for it to compute) and then try different dropout values (between 0,1). Whatever 
model has the best validation performance (the loss, written in the checkpoint 
filename, low is good) is the one you should use in the end.

It is very common in deep learning to run many different models with many 
different hyperparameter settings, and in the end take whatever checkpoint 
gave the best validation performance.

By the way, the size of your training and validation splits are also parameters. 
Make sure you have a decent amount of data in your validation set or otherwise 
the validation performance will be noisy and not very informative.

"""

tf.reset_default_graph()
with open('anna.txt', 'r') as f:
    text = f.read()

vocab = sorted(set(text))
vocab_to_int = {c:i for i, c in enumerate(vocab)}
int_to_vocab = dict(enumerate(vocab))
encoded = np.array([vocab_to_int[c] for c in text])

batch_size = 100        # Sequences per batch
num_steps = 100         # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
num_layers = 2          # Number of LSTM layers
learning_rate = 0.001   # Learning rate
keep_prob = 0.5         # Dropout keep probability

epochs = 20
# Print losses every N interations
print_every_n = 2

# Save every N iterations
save_every_n = 200

model = CharRNN(num_classes=len(vocab), batch_size=batch_size, 
                num_steps = num_steps, lstm_size = lstm_size,
                num_layers = num_layers, learning_rate = learning_rate)

saver = tf.train.Saver(max_to_keep=100)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    counter = 0
    for e in range(epochs):
        new_state = sess.run(model.initial_state)
        loss = 0
        batch_num = 0
        for x,y in get_batches(encoded, batch_size, num_steps):
            counter +=1
            batch_num +=1
            start = time.time()
            feed = {model.inputs:x,
                    model.targets:y,
                    model.keep_prob:keep_prob,
                    model.initial_state:new_state}
            batch_loss, new_state, _ = sess.run([model.loss, 
                                                 model.final_state, 
                                                 model.optimizer], 
                                                 feed_dict=feed)
            if (counter % print_every_n == 0):
                end = time.time()
                print('Epoch: {}/{}... '.format(e+1, epochs),
                      'Batch num : {}... '.format(batch_num),
                      'Training Step: {}... '.format(counter),
                      'Training loss: {:.4f}... '.format(batch_loss),
                      '{:.4f} sec/batch'.format((end-start)))
        
            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
            if (counter % save_every_n == 0):
                saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))
    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))