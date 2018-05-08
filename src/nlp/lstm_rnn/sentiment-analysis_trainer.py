# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 15:28:46 2018

@author: Samriddha.Chatterjee
"""

import numpy as np
import tensorflow as tf
from string import punctuation
from collections import Counter

lstm_size = 250
lstm_layers = 1
batch_size = 500
learnig_rate = 0.01
embed_size = 300
epochs = 10

def pre_process_data(reviews, labels):
    all_text = ''.join([c for c in reviews if c not in punctuation])
    reviews = all_text.split('\n')
    all_text = ' '.join(reviews)
    words = all_text.split()
    counts = Counter(words)
    vocab = sorted(counts, key=counts.get, reverse=True)
    vocab_to_int = {w: i for i, w in enumerate(vocab)}
    reviews_int = []
    for each in reviews:
        reviews_int.append([vocab_to_int[word] for word in each.split()])
    labels = labels.split('\n')
    labels = np.array([1 if each == 'positive' else 0 for each in labels])
    non_zero_idx = [ii for ii, reviews in enumerate(reviews_int) if len(reviews)!=0]
    
    reviews_int = [reviews_int[ii] for ii in non_zero_idx]
    labels = np.array([labels[ii] for ii in non_zero_idx])
    seq_len = 200
    features = np.zeros((len(reviews_int), seq_len), dtype='int32')
    
    for i, row in enumerate(reviews_int):
        features[i,-len(row):] = np.array(row)[:seq_len]
        
    return features, labels, vocab_to_int
    
 
def train_val_test_split(features, labels):
    
    split_idx = int(len(features)*0.8)
    train_x, val_x = features[:split_idx], features[split_idx:]
    train_y, val_y = labels[:split_idx], labels[split_idx:]
    
    test_split_idx = int(len(val_x)*0.5)
    val_x, test_x = features[:test_split_idx], features[test_split_idx:]
    val_y, test_y = labels[:test_split_idx], labels[test_split_idx:]
    
    return train_x, train_y, val_x, val_y, test_x, test_y
    
    
def get_batches(x, y, batch_size):
    n_batches = len(x)//batch_size
    x, y = x[:n_batches*batch_size], y[:n_batches*batch_size]
    for ii in range(0, len(x), batch_size):
        yield x[ii:ii+batch_size], y[ii:ii+batch_size] 
    
def get_graph():
    return tf.Graph()

def build_graph(graph, vocab_to_int):
    n_words = len(vocab_to_int)+1
    with graph.as_default():
        input_ = tf.placeholder(tf.int32, [None, None], name='inputs')
        label_ = tf.placeholder(tf.int32, [None, None], name='labels')
        keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        embedding = tf.Variable(tf.random_uniform((n_words, embed_size), -1, 1))
        embed = tf.nn.embedding_lookup(embedding, input_)
        
        lstm = tf.contrib.rnn.BasicLSTMCell(lstm_size)
        drop = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        cell = tf.contrib.rnn.MultiRNNCell([drop]*lstm_layers)
        initial_state = cell.zero_state(batch_size, tf.float32)
        
        outputs, final_state = tf.nn.dynamic_rnn(cell, embed, initial_state=initial_state)
        predictions = tf.contrib.layers.fully_connected(outputs[:,-1], 1, activation_fn=tf.sigmoid)
        cost = tf.losses.mean_squared_error(label_, predictions)
        optimizer = tf.train.AdamOptimizer(learnig_rate).minimize(cost)
        
        correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), label_)
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
        
    return input_, label_, keep_prob, cell, initial_state, cost, final_state, optimizer, accuracy
        
        
        

def train_test(flag):
    with open('reviews.txt', 'r') as f:
        reviews = f.read()
    with open('labels.txt', 'r') as f:
        labels = f.read()
    features, labels, vocab_to_int = pre_process_data(reviews, labels)
    train_x, train_y, val_x, val_y, test_x, test_y = train_val_test_split(features, labels)
    graph = get_graph()
    input_, label_, keep_prob, cell, initial_state, cost, final_state, optimizer, accuracy = build_graph(graph, vocab_to_int)
    saver = tf.train.Saver()
    if flag == 'train':
        with graph.as_default():
            with tf.Session() as session:
                session.run(tf.global_variables_initializer())
                iteration = 1
                for e in range(epochs):
                    state = session.run(initial_state)
                    for ii, (x,y) in enumerate(get_batches(train_x, train_y, batch_size), 1):
                        feed = {
                                input_:x,
                                label_:y[:,None],
                                keep_prob:0.5,
                                initial_state:state
                                }
                        loss, state, _ = session.run([cost, final_state, optimizer], feed_dict=feed)
                        if iteration%2 ==0:
                            print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))
                        if iteration%25==0:
                            val_acc = []
                            val_state = session.run(cell.zero_state(batch_size, tf.float32))
                            for x, y in get_batches(val_x, val_y, batch_size):
                                feed = {input_: x,
                                        label_: y[:, None],
                                        keep_prob: 1,
                                        initial_state: val_state}
                                batch_acc, val_state = session.run([accuracy, final_state], feed_dict=feed)
                                val_acc.append(batch_acc)
                                print("Val acc: {:.3f}".format(np.mean(val_acc)))
                        iteration +=1
                saver.save(session, 'checkpoints/sentiment.ckpt')
    else:
        test_acc = []
        with tf.Session(graph=graph) as session:
            saver.restore(session, tf.train.latest_checkpoint('checkpoints'))
            test_state = session.run(cell.zero_state(batch_size, tf.float32))
            for ii, (x,y) in enumerate(get_batches(test_x, test_y, batch_size), 1):
                feed = {
                        input_:x,
                        label_:y[:,None],
                        keep_prob:1.0,
                        initial_state:test_state
                        }
                batch_acc, test_state = session.run([accuracy, final_state], feed_dict=feed)
                test_acc.append(batch_acc)
            print('Test accuracy : ', np.mean(test_acc))
        
if __name__=='__main__':
    train_test('train')    