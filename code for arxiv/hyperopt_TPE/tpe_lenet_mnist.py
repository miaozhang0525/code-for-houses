#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 21:24:59 2018

@author: mzhang3
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 14:18:37 2018

@author: mzhang3
"""

import numpy as np
import time
import os

import tensorlayer as tl
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pyDOE import *
from hyperopt import fmin, tpe, hp, rand,space_eval

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28,28))

space = [hp.uniform('conv1', 10,70),hp.uniform('conv2', 10,70),hp.uniform('drop_out_rate1', 30,70),
         hp.uniform('dense1', 200,400),hp.uniform('drop_out_rate2', 30,70),hp.uniform('dense2', 100,200),hp.uniform('drop_out_rate3', 30,70)]



def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

y_train=convert_to_one_hot(y_train,10)
y_val=convert_to_one_hot(y_val,10)
y_test=convert_to_one_hot(y_test,10)

x_final_test=X_test
y_final_test=y_test



x_train = np.vstack((X_train, X_val))
y_train =np.vstack((y_train, y_val))

     # x upper and lower bounds
n_val=5#####number of validation




def get_reward_test(x_train, y_train, x_test, y_test, hp_setting):
    test_error=0
    for i in range(n_val):            
     
        tf.reset_default_graph()
        batch_size = 500
        sess = tf.InteractiveSession()              
        conv1=hp_setting[0]
        conv2=hp_setting[1]
        drop_out_rate1=hp_setting[2]*0.01
        dense1=hp_setting[3]
        drop_out_rate2=hp_setting[4]*0.01
        dense2=hp_setting[5]
        drop_out_rate3=hp_setting[6]*0.01
        
        x = tf.placeholder(tf.float32, shape=[None, 28,28], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')        
        #architecture
        data_in = tl.layers.InputLayer(x, name='input_layer')
        
        data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
        data_in = tl.layers.ExpandDimsLayer(data_in,axis=3,name='add')
        net = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [3, 3, 1, conv1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_1')
        
        net = tl.layers.BatchNormLayer(net,is_train=True, act=tf.nn.relu, name='bn1_1')
        
        net = tl.layers.PoolLayer(net,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_1',)
        
        net = tl.layers.Conv2dLayer(net,
                                   act = tf.identity,
                                   shape = [3, 3, conv1,conv2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_2')
        
        net = tl.layers.BatchNormLayer(net,is_train=True, act=tf.nn.relu, name='bn1')
        net = tl.layers.PoolLayer(net,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_2',)
        
        net = tl.layers.FlattenLayer(net,name='flatten_layer1')
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate1, name='drop1_1')
        
        net= tl.layers.DenseLayer(net,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate2, name='drop1')
        net= tl.layers.DenseLayer(net,
                                  n_units = dense2,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer2',)
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate3, name='drop2')
            
        out = tl.layers.DenseLayer(net, n_units=10,
                                         act = tf.identity,
                                         name='output_layer')
        
        
        
        y = out.outputs
    
        
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
        #tf.summary.scalar(name='loss',tensor = cross_entropy)
        #training
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        # correct_prediction = tf.equal(y_conv, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        train_params = out.all_params
        
        train_op = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999,
                                    epsilon=1e-08, use_locking=False)\
            .minimize(cross_entropy, var_list=train_params)
        
        
        tl.layers.initialize_global_variables(sess)
        
        out.print_params()
        out.print_layers()
        
        tl.utils.fit(sess, out, train_op, cross_entropy, x_train, y_train, x, y_,
                    acc=accuracy, batch_size=50, n_epoch=100, print_freq=1,
                    X_val=x_test, y_val=y_test, eval_train=True,)
        
        y_op = tf.argmax(tf.nn.softmax(y), 1)       
        y_test_val=sess.run(tf.argmax(y_test, 1))
        pre_index = 0
        validation_acc = 0.0
        iteration=int(y_test.shape[0]/batch_size)
        
        for step in range(0, iteration):
            if pre_index+batch_size <= y_test.shape[0] :
                batch_x = x_test[pre_index : pre_index+batch_size]
                batch_y = y_test_val[pre_index : pre_index+batch_size]
            else :
                batch_x = x_test[pre_index : ]
                batch_y = y_test_val[pre_index : ]
        
            y_prediction=tl.utils.predict(sess, out, batch_x, x, y_op)
            c_mat, f1, batch_acc, f1_macro = tl.utils.evaluation(batch_y, y_prediction, 10)
            validation_acc += batch_acc
            pre_index += batch_size
        validation_acc /= iteration
        test_error=test_error+(1-validation_acc)
    sess.close()
   
    return test_error/n_val


def get_reward(args):
# Hyper Parameters   
    
    
    X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28,28))
    
    y_train=convert_to_one_hot(y_train,10)
    y_val=convert_to_one_hot(y_val,10)
    y_test=convert_to_one_hot(y_test,10)
    
    x_final_test=X_test
    y_final_test=y_test
    
    
    x_train = np.vstack((X_train, X_val))
    y_train =np.vstack((y_train, y_val))
    
    cross_validation_acc=0
       
    conv1,conv2,drop_out_rate1,dense1,drop_out_rate2,dense2,drop_out_rate3 = args

    conv1=np.floor(conv1)
    conv2=np.floor(conv2)
    drop_out_rate1=drop_out_rate1*0.01
    dense1=np.floor(dense1)
    drop_out_rate2=drop_out_rate2*0.01
    dense2=np.floor(dense2)
    drop_out_rate3=drop_out_rate3*0.01
    
    
    cross_validation_acc=0
    
    for pp in range(n_val):  
            
        permutation=np.random.permutation(y_train.shape[0])   
        shuffled_data=x_train[permutation,]
        shuffled_labels=y_train[permutation,]  
        
        a=int(y_train.shape[0]*3/4)
        x_train=shuffled_data[range(a),:]
        y_train=shuffled_labels[range(a),]       
        x_test=shuffled_data[a:shuffled_data.shape[0],:]
        y_test=shuffled_labels[a:shuffled_data.shape[0],]   
        
        
        tf.reset_default_graph()
        batch_size = 200
        sess = tf.InteractiveSession()       
        

        
        x = tf.placeholder(tf.float32, shape=[None, 28,28], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')        
        #architecture
        data_in = tl.layers.InputLayer(x, name='input_layer')
        
        data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
        data_in = tl.layers.ExpandDimsLayer(data_in,axis=3,name='add')
        net = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [3, 3, 1, conv1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_1')
        
        net = tl.layers.BatchNormLayer(net,is_train=True, act=tf.nn.relu, name='bn1_1')
        
        net = tl.layers.PoolLayer(net,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_1',)
        
        net = tl.layers.Conv2dLayer(net,
                                   act = tf.identity,
                                   shape = [3, 3, conv1,conv2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_2')
        
        net = tl.layers.BatchNormLayer(net,is_train=True, act=tf.nn.relu, name='bn1')
        net = tl.layers.PoolLayer(net,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_2',)
        
        net = tl.layers.FlattenLayer(net,name='flatten_layer1')
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate1, name='drop1_1')
        
        net= tl.layers.DenseLayer(net,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate2, name='drop1')
        net= tl.layers.DenseLayer(net,
                                  n_units = dense2,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer2',)
        net = tl.layers.DropoutLayer(net, keep=drop_out_rate3, name='drop2')
            
        out = tl.layers.DenseLayer(net, n_units=10,
                                         act = tf.identity,
                                         name='output_layer')
        
                
        y = out.outputs
        

        
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
        #tf.summary.scalar(name='loss',tensor = cross_entropy)
        #training
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        # correct_prediction = tf.equal(y_conv, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
        train_params = out.all_params
        
        train_op = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999,
                                    epsilon=1e-08, use_locking=False)\
            .minimize(cross_entropy, var_list=train_params)
        
        
        tl.layers.initialize_global_variables(sess)
        
        out.print_params()
        out.print_layers()
        
        tl.utils.fit(sess, out, train_op, cross_entropy, x_train, y_train, x, y_,
                    acc=accuracy, batch_size=50, n_epoch=100, print_freq=1,
                    X_val=x_test, y_val=y_test, eval_train=True,)
        
        y_op = tf.argmax(tf.nn.softmax(y), 1)       
        y_test_val=sess.run(tf.argmax(y_test, 1))
        pre_index = 0
        validation_acc = 0.0
        iteration=int(y_test.shape[0]/batch_size)
        
        for step in range(0, iteration):
            if pre_index+batch_size <= y_test.shape[0] :
                batch_x = x_test[pre_index : pre_index+batch_size]
                batch_y = y_test_val[pre_index : pre_index+batch_size]
            else :
                batch_x = x_test[pre_index : ]
                batch_y = y_test_val[pre_index : ]
        
            y_prediction=tl.utils.predict(sess, out, batch_x, x, y_op)
            c_mat, f1, batch_acc, f1_macro = tl.utils.evaluation(batch_y, y_prediction, 10)
            validation_acc += batch_acc
            pre_index += batch_size
        validation_acc /= iteration
        cross_validation_acc+=validation_acc
        sess.close()
    cross_validation_acc/=n_val    
    return 1-cross_validation_acc



best = fmin(get_reward, space, algo=tpe.suggest,max_evals=(200/n_val))

select_hp=[np.floor(best['conv1']),np.floor(best['conv2']),best['drop_out_rate1'],np.floor(best['dense1']),best['drop_out_rate2'],np.floor(best['dense2']),best['drop_out_rate3']]

test_error=get_reward_test(x_train, y_train, x_final_test, y_final_test, select_hp)


np.savetxt('pop_reserve_tpe_lenet_mnist.txt',select_hp,delimiter=',')    

np.save('test_error_tpe_lenet_mnist.npy',test_error) 
