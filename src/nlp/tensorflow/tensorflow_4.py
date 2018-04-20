# -*- coding: utf-8 -*-
"""
Created on Mon Apr 16 17:18:10 2018

@author: Samriddha.Chatterjee
"""
from urllib.request import urlretrieve
from zipfile import ZipFile
from tqdm import tqdm
from PIL import Image
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import os
import hashlib
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import math

def download(url, file):
    if not os.path.isfile(file):
        print("Downloading {} file..".format(file))
        urlretrieve(url,file)
        print("Downloaded {} file..".format(file))

def download_data():
    download('https://s3.amazonaws.com/udacity-sdc/notMNIST_train.zip', 'notMNIST_train.zip')
    download('https://s3.amazonaws.com/udacity-sdc/notMNIST_test.zip', 'notMNIST_test.zip')
    assert hashlib.md5(open('notMNIST_train.zip', 'rb').read()).hexdigest() == 'c8673b3f28f489e9cdf3a3d74e2ac8fa', \
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
    assert hashlib.md5(open('notMNIST_test.zip', 'rb').read()).hexdigest() == '5d3c7e653e63471c88df796156a9dfa9', \
        'notMNIST_train.zip file is corrupted.  Remove the file and try again.'
    print('All files downloaded.')


def uncompress_feature_labels(file):
    features = []
    labels = []
    with ZipFile(file) as zfile:
        filenames_pbar = tqdm(zfile.namelist(), unit='files')
        for filename in filenames_pbar:
            if not filename.endswith('/'):
                with zfile.open(filename) as imageFile:
                    image = Image.open(imageFile)
                    image.load()
                    feature = np.array(image, dtype='float32').flatten()
                label = os.path.split(filename)[0][1]
                labels.append(label)
                features.append(feature)
    return np.array(features), np.array(labels)

def normalize_grayscale(image_data):
    a = 0.1
    b = 0.9
    x_min = 0
    x_max = 255
    return a + ((image_data - x_min)*(b-a)/(x_max-x_min))

def run():
    download_data()
    train_features, train_labels = uncompress_feature_labels('notMNIST_train.zip')
    test_features, test_labels = uncompress_feature_labels('notMNIST_test.zip')
    size_limit = 150000
    train_features, train_labels = resample(train_features, train_labels, n_samples=size_limit)
    
    train_features = normalize_grayscale(train_features)
    test_features = normalize_grayscale(test_features)
    
    encoder = LabelBinarizer()
    encoder.fit(train_labels)
    train_labels = encoder.transform(train_labels)
    test_labels = encoder.transform(test_labels)
    train_labels = train_labels.astype(np.float32)
    test_labels = test_labels.astype(np.float32)
    
    train_features, valid_features, train_labels, valid_labels = train_test_split(
                                                                    train_features,
                                                                    train_labels,
                                                                    test_size=0.05,
                                                                    random_state=832289)
    
    pickle_file = 'notMNIST.pickle'
    if not os.path.isfile(pickle_file):
        print('Saving data to pickle file...')
        try:
            with open('notMNIST.pickle', 'wb') as pfile:
                pickle.dump(
                    {
                        'train_dataset': train_features,
                        'train_labels': train_labels,
                        'valid_dataset': valid_features,
                        'valid_labels': valid_labels,
                        'test_dataset': test_features,
                        'test_labels': test_labels,
                    },
                    pfile, pickle.HIGHEST_PROTOCOL)
        except Exception as e:
            print('Unable to save data to', pickle_file, ':', e)
            raise
    features_count = 784
    labels_count = 10
    
    features = tf.placeholder(tf.float32)
    labels = tf.placeholder(tf.float32)
    weights = tf.Variable(tf.truncated_normal((features_count, labels_count)))
    biases = tf.Variable(tf.zeros(labels_count))
    
    # Feed dicts for training, validation, and test session
    train_feed_dict = {features: train_features, labels: train_labels}
    valid_feed_dict = {features: valid_features, labels: valid_labels}
    test_feed_dict = {features: test_features, labels: test_labels}
    
    # Linear Function WX + b
    logits = tf.matmul(features, weights) + biases
    
    prediction = tf.nn.softmax(logits)
    
    # Cross entropy
    cross_entropy = -tf.reduce_sum(labels * tf.log(prediction), reduction_indices=1)
    
    # Training loss
    loss = tf.reduce_mean(cross_entropy)
    
    # Determine if the predictions are correct
    is_correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1))
    # Calculate the accuracy of the predictions
    accuracy = tf.reduce_mean(tf.cast(is_correct_prediction, tf.float32))
    
    print('Accuracy function created.')
    batch_size = 128
    epochs = 1 
    learning_rate = 0.1
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    validation_accuracy = 0.0
    log_batch_step = 50
    batches = []
    loss_batch = []
    train_acc_batch = []
    valid_acc_batch = []
    # Create an operation that initializes all variables
    init = tf.global_variables_initializer()
    with tf.Session() as session:
        session.run(init)
        session.run(loss, feed_dict=train_feed_dict)
        session.run(loss, feed_dict=valid_feed_dict)
        session.run(loss, feed_dict=test_feed_dict)
        biases_data = session.run(biases)
        
        batch_count = int(math.ceil(len(train_features)/batch_size))

        for epoch_i in range(epochs):
            
            # Progress bar
            batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(epoch_i+1, epochs), unit='batches')
            
            # The training cycle
            for batch_i in batches_pbar:
                # Get a batch of training features and labels
                batch_start = batch_i*batch_size
                batch_features = train_features[batch_start:batch_start + batch_size]
                batch_labels = train_labels[batch_start:batch_start + batch_size]
    
                # Run optimizer and get loss
                _, l = session.run(
                    [optimizer, loss],
                    feed_dict={features: batch_features, labels: batch_labels})
                print('Loss ', l)
                # Log every 50 batches
                if not batch_i % log_batch_step:
                    # Calculate Training and Validation accuracy
                    training_accuracy = session.run(accuracy, feed_dict=train_feed_dict)
                    validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
    
                    # Log batches
                    previous_batch = batches[-1] if batches else 0
                    batches.append(log_batch_step + previous_batch)
                    loss_batch.append(l)
                    train_acc_batch.append(training_accuracy)
                    valid_acc_batch.append(validation_accuracy)
    
            # Check accuracy against Validation data
            validation_accuracy = session.run(accuracy, feed_dict=valid_feed_dict)
            print('Validation accuracy at {}'.format(validation_accuracy))
        loss_plot = plt.subplot(211)
        loss_plot.set_title('Loss')
        loss_plot.plot(batches, loss_batch, 'g')
        loss_plot.set_xlim([batches[0], batches[-1]])
        acc_plot = plt.subplot(212)
        acc_plot.set_title('Accuracy')
        acc_plot.plot(batches, train_acc_batch, 'r', label='Training Accuracy')
        acc_plot.plot(batches, valid_acc_batch, 'x', label='Validation Accuracy')
        acc_plot.set_ylim([0, 1.0])
        acc_plot.set_xlim([batches[0], batches[-1]])
        acc_plot.legend(loc=4)
        plt.tight_layout()
        plt.show()
        test_accuracy = session.run(accuracy, feed_dict=test_feed_dict)
        print('Nice Job! Test Accuracy is {}'.format(test_accuracy))


if __name__=='__main__':
    run()
    