# -*- coding: utf-8 -*-
"""
Created on Sat Apr 14 21:12:17 2018

@author: Samriddha.Chatterjee
"""

import tensorflow as tf

hello_constant = tf.constant('Hello tensorflow')
x = tf.placeholder(tf.string)
y = tf.placeholder(tf.int32)
z = tf.placeholder(tf.float32)

x_1 = tf.add(tf.constant(5), tf.constant(2))
x_2 = tf.subtract(tf.constant(10), tf.constant(4)) # 6
x_3 = tf.multiply(tf.constant(2), tf.constant(5))  # 10
x_1 = x_3

x_4 = tf.divide(tf.cast(tf.constant(5.0), tf.int32), tf.constant(2))

with tf.Session() as session:
    output = session.run(hello_constant)
    print(output)
    output_1 = session.run(x, feed_dict={x:'Hello', y:23, z:8.09})
    print(output_1)
    output_2 = session.run(x, feed_dict={x:'Hello_0', y:23, z:8.09})
    print(output_2)
    
    o_1 = session.run(x_1)
    print(o_1)
    o_2 = session.run(x_2)
    print(o_2)
    o_3 = session.run(x_3)
    print(o_3)
    o_4 = session.run(x_4)
    print(o_4)
    
    