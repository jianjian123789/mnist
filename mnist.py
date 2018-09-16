# coding:utf-8
# 1.准备
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import sys
import argparse
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def selu(x):
    with tf.name_scope('elu') as scope:
        alpha=1.67
        scale=1.05
        return scale*tf.where(x>=0.0,x,alpha*tf.nn.elu(x))
    
def tanh(x):
    return tf.nn.tanh(x)

def activation(x):
#     return relu(x)
    return selu(x)

def initialize(shape,stddev=0.1):
    return tf.truncated_normal(shape,stddev=stddev)
    

data_dir='./mnist_data/'
mnist=input_data.read_data_sets(data_dir,one_hot=True)

# 2.前传：输入（值、轮数）、权重、输出
x =tf.placeholder(tf.float32,[None,784])
y_=tf.placeholder(tf.float32,[None,10])

init_learning_rate=tf.placeholder(tf.float32)
epoch_steps=tf.to_int64(tf.div(60000,tf.shape(x)[0]))
global_step=tf.train.get_or_create_global_step()
current_epoch=global_step//epoch_steps
decay_times=current_epoch
current_learning_rate=tf.multiply(init_learning_rate,tf.pow(0.575,tf.to_float(decay_times)))


L1_units_count=100
w1=tf.Variable(initialize([784,L1_units_count],stddev=np.sqrt(2/784)))
b1=tf.Variable(tf.constant(0.001,shape=[L1_units_count]))
logits1=tf.matmul(x,w1)+b1
output1=activation(logits1)

L2_units_count=10
w2=tf.Variable(initialize([L1_units_count,L2_units_count],stddev=np.sqrt(2/L1_units_count)))
b2=tf.Variable(tf.constant(0.001,shape=[L2_units_count]))
logits2=tf.matmul(output1,w2)+b2
y=logits2


# 3.反传：损失函数和反向传播优化算法
cross_entropy=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=y))
l2_loss=tf.nn.l2_loss(w1)+tf.nn.l2_loss(w2)
total_loss=cross_entropy+4e-5*l2_loss
train_step=tf.train.AdamOptimizer(current_learning_rate).minimize(total_loss,global_step=global_step)



correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# 4.迭代
sess=tf.Session()
init_op=tf.global_variables_initializer()
sess.run(init_op)


for step in range(3000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  lr = 1e-2
  _, loss, l2_loss_value, total_loss_value, current_lr_value = \
      sess.run(
               [train_step, cross_entropy, l2_loss, total_loss, 
                current_learning_rate], 
               feed_dict={x: batch_xs, y_: batch_ys, 
                          init_learning_rate:lr})
  
  if (step+1) % 100 == 0:
    print('step %d, entropy loss: %f, l2_loss: %f, total loss: %f' % 
            (step+1, loss, l2_loss_value, total_loss_value))
    #print(sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys}))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images,
                                    y_: mnist.test.labels}))
    #print(current_lr_value)


