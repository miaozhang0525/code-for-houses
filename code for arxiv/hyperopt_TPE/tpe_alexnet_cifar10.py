#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 15:53:19 2018

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

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32,32,3),plotable=False)

space = [hp.uniform('conv1', 10,70),hp.uniform('conv2', 10,70),hp.uniform('conv3', 10,70),hp.uniform('conv4', 10,70),
         hp.uniform('conv5', 10,70),hp.uniform('dense1', 200,400),hp.uniform('drop_out_rate1', 30,70),hp.uniform('dense2', 100,200),hp.uniform('drop_out_rate2', 30,70)]

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

y_train=convert_to_one_hot(y_train,10)

y_test=convert_to_one_hot(y_test,10)

x_final_test=X_test
y_final_test=y_test



x_train = X_train


DNA_SIZE = 9            # DNA length
POP_SIZE = 2           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.2    # mutation probability
N_GENERATIONS = 2
x_bound = np.array([[10, 10,30,100,30],[70,70,70,200,70]])      # x upper and lower bounds
n_val=5#####number of validation




def get_reward_test(x_train, y_train, x_test, y_test, hp_setting):     
    
    test_error=0
    for it in range(n_val):      
        tf.reset_default_graph()
                 
        batch_size = 200
        sess = tf.InteractiveSession()              
        conv1=hp_setting[0]
        conv2=hp_setting[1]
        conv1=hp_setting[2]
        conv2=hp_setting[3]
        conv2=hp_setting[4]
        dense1=hp_setting[5]
        drop_out_rate1=hp_setting[6]*0.01
        dense2=hp_setting[7]
        drop_out_rate2=hp_setting[8]*0.01
    
    
        
                
        tf.reset_default_graph()
        sess = tf.InteractiveSession()
            
        x = tf.placeholder(tf.float32, shape=[None, 32, 32,3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')
        
        #architecture
        data_in = tl.layers.InputLayer(x, name='input_layer')
        
        data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
        c1 = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [3, 3, 3, conv1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1')
        bn1 = tl.layers.BatchNormLayer(c1,is_train=True, act=tf.nn.relu, name='bn1')
        p1 = tl.layers.PoolLayer(bn1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1',)
        c2 = tl.layers.Conv2dLayer(p1,
                                   act = tf.identity,
                                   shape = [3, 3, conv1,conv2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer2')
        bn2 = tl.layers.BatchNormLayer(c2,is_train=True, act=tf.nn.relu, name='bn2')
        p2 = tl.layers.PoolLayer(bn2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer2',)
        c3 = tl.layers.Conv2dLayer(p2,
                                   act = tf.identity,
                                   shape = [3, 3, conv2,conv3],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer3')
        bn3 = tl.layers.BatchNormLayer(c3,is_train=True, act=tf.nn.relu, name='bn3')
     
        c4 = tl.layers.Conv2dLayer(bn3,
                                   act = tf.identity,
                                   shape = [3, 3, conv3,conv4],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer4')
        bn4 = tl.layers.BatchNormLayer(c4,is_train=True, act=tf.nn.relu, name='bn4')
          
        c5 = tl.layers.Conv2dLayer(bn4,
                                   act = tf.identity,
                                   shape = [3, 3, conv4,conv5],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer5')
        bn5 = tl.layers.BatchNormLayer(c5,is_train=True, act=tf.nn.relu, name='bn5')
        p5 = tl.layers.PoolLayer(bn5,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer5',)    
        
        
         
        fla1 = tl.layers.FlattenLayer(p5,name='flatten_layer1')
    
        f1= tl.layers.DenseLayer(fla1,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        drop1= tl.layers.DropoutLayer(f1, keep=drop_out_rate1, name='drop1')
        f2= tl.layers.DenseLayer(drop1,
                                  n_units = dense2,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer2',)
        drop2= tl.layers.DropoutLayer(f2, keep=drop_out_rate2, name='drop2')    
        out = tl.layers.DenseLayer(drop2, n_units=10,
                                         act = tf.identity,
                                         name='output_layer')
         
        y = out.outputs
        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
        
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        # correct_prediction = tf.equal(y_conv, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.summary.scalar('accuracy',accuracy)
        # optimizer
        train_params = out.all_params
        
        train_op = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999,
                                    epsilon=1e-08, use_locking=False)\
            .minimize(cross_entropy, var_list=train_params)
        
        
        tl.layers.initialize_global_variables(sess)
        
        out.print_params()
        out.print_layers()
        
        # 
        tl.utils.fit(sess, out, train_op, cross_entropy, x_train, y_train, x, y_,
                    acc=accuracy, batch_size=200, n_epoch=2, print_freq=10,
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
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32,32,3),plotable=False)
    
    y_train=convert_to_one_hot(y_train,10)
       
    x_train = X_train
    
    cross_validation_acc=0
       
    conv1,conv2, conv3,conv4, conv5,dense1,drop_out_rate1,dense2,drop_out_rate2 = args
    
    conv1=np.floor(conv1)
    conv2=np.floor(conv2) 
    conv3=np.floor(conv3)
    conv4=np.floor(conv4)     
    conv5=np.floor(conv5)
    dense1=np.floor(dense1)
    drop_out_rate1=drop_out_rate1*0.01
    dense2=np.floor(dense2)
    drop_out_rate2=drop_out_rate2*0.01
    
    
    cross_validation_acc=0
    batch_size = 200
    
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

            
        x = tf.placeholder(tf.float32, shape=[None, 32, 32,3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')
        
        #architecture
        data_in = tl.layers.InputLayer(x, name='input_layer')
        
        data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
        c1 = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [3, 3, 3, conv1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1')
        bn1 = tl.layers.BatchNormLayer(c1,is_train=True, act=tf.nn.relu, name='bn1')
        p1 = tl.layers.PoolLayer(bn1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1',)
        c2 = tl.layers.Conv2dLayer(p1,
                                   act = tf.identity,
                                   shape = [3, 3, conv1,conv2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer2')
        bn2 = tl.layers.BatchNormLayer(c2,is_train=True, act=tf.nn.relu, name='bn2')
        p2 = tl.layers.PoolLayer(bn2,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer2',)
        c3 = tl.layers.Conv2dLayer(p2,
                                   act = tf.identity,
                                   shape = [3, 3, conv2,conv3],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer3')
        bn3 = tl.layers.BatchNormLayer(c3,is_train=True, act=tf.nn.relu, name='bn3')
     
        c4 = tl.layers.Conv2dLayer(bn3,
                                   act = tf.identity,
                                   shape = [3, 3, conv3,conv4],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer4')
        bn4 = tl.layers.BatchNormLayer(c4,is_train=True, act=tf.nn.relu, name='bn4')
          
        c5 = tl.layers.Conv2dLayer(bn4,
                                   act = tf.identity,
                                   shape = [3, 3, conv4,conv5],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer5')
        bn5 = tl.layers.BatchNormLayer(c5,is_train=True, act=tf.nn.relu, name='bn5')
        p5 = tl.layers.PoolLayer(bn5,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer5',)    
        
        
         
        fla1 = tl.layers.FlattenLayer(p5,name='flatten_layer1')
    
        f1= tl.layers.DenseLayer(fla1,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        drop1= tl.layers.DropoutLayer(f1, keep=drop_out_rate1, name='drop1')
        f2= tl.layers.DenseLayer(drop1,
                                  n_units = dense2,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer2',)
        drop2= tl.layers.DropoutLayer(f2, keep=drop_out_rate2, name='drop2')    
        out = tl.layers.DenseLayer(drop2, n_units=10,
                                         act = tf.identity,
                                         name='output_layer')
         
        y = out.outputs

        cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_))
        
        
        correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
        # correct_prediction = tf.equal(y_conv, y_)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #tf.summary.scalar('accuracy',accuracy)
        # optimizer
        train_params = out.all_params
        
        train_op = tf.train.AdamOptimizer(learning_rate=0.0005, beta1=0.9, beta2=0.999,
                                    epsilon=1e-08, use_locking=False)\
            .minimize(cross_entropy, var_list=train_params)
        
        
        tl.layers.initialize_global_variables(sess)
        
        out.print_params()
        out.print_layers()
        
        # 
        tl.utils.fit(sess, out, train_op, cross_entropy, x_train, y_train, x, y_,
                    acc=accuracy, batch_size=200, n_epoch=2, print_freq=10,
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



best = fmin(get_reward, space, algo=tpe.suggest,max_evals=200/n_val)

select_hp=[np.floor(best['conv1']),np.floor(best['conv2']),best['drop_out_rate1'],np.floor(best['dense1']),best['drop_out_rate2']]

test_error=get_reward_test(x_train, y_train, x_final_test, y_final_test, select_hp)


np.savetxt('pop_reserve_tpe_alexnet_cifar10.txt',select_hp,delimiter=',')    

np.save('test_error_tpe_alexnet_cifar10.npy',test_error) 