#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 17:20:25 2018

@author: mzhang3
"""

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

space = [hp.uniform('conv1_1',10,70),hp.uniform('conv1_2', 10,70),hp.uniform('conv2_1', 10,70),
         hp.uniform('conv2_2', 10,70),hp.uniform('conv3_1', 10,70),hp.uniform('conv3_2', 10,70),
         hp.uniform('drop_out_rate1', 30,70),hp.uniform('dense1', 200,400),hp.uniform('drop_out_rate2', 30,70)]

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

y_train=convert_to_one_hot(y_train,10)

y_test=convert_to_one_hot(y_test,10)

x_final_test=X_test
y_final_test=y_test



x_train = X_train


n_val=5#####number of validation




def get_reward_test(x_train, y_train, x_test, y_test, hp_setting):     
    
    
    tf.reset_default_graph()
    batch_size=200
                          
    conv1_1=hp_setting[0]
    conv1_2=hp_setting[1]
    conv2_1=hp_setting[2]
    conv2_2=hp_setting[3]    
    conv3_1=hp_setting[4]
    conv3_2=hp_setting[5]        
    drop_out_rate1=hp_setting[6]*0.01
    dense1=hp_setting[7]
    drop_out_rate2=hp_setting[8]*0.01
    
            
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
        
    x = tf.placeholder(tf.float32, shape=[None, 32, 32,3], name='x')
    y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')
    
    #architecture
    data_in = tl.layers.InputLayer(x, name='input_layer')
    
    data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
    c1_1 = tl.layers.Conv2dLayer(data_in,
                               act = tf.identity,
                               shape = [3, 3, 3, conv1_1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer1_1')
    bn1_1 = tl.layers.BatchNormLayer(c1_1,is_train=True, act=tf.nn.relu, name='bn1_1')
    p1_1 = tl.layers.PoolLayer(bn1_1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer1_1',)
    c1_2 = tl.layers.Conv2dLayer(p1_1,
                               act = tf.identity,
                               shape = [3, 3, conv1_1,conv1_2],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer1_2')
    bn1 = tl.layers.BatchNormLayer(c1_2,is_train=True, act=tf.nn.relu, name='bn1')
    p1 = tl.layers.PoolLayer(bn1,
                             ksize=[1, 7, 7, 1],
                             strides=[1, 7, 7, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer1_2',)
         
    c2_1 = tl.layers.Conv2dLayer(data_in,
                               act = tf.identity,
                               shape = [5, 5, 3, conv2_1],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer2_1')
    bn2_1 = tl.layers.BatchNormLayer(c2_1,is_train=True, act=tf.nn.relu, name='bn2_1')
    p2_1 = tl.layers.PoolLayer(bn2_1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer2_1',)
    c2_2 = tl.layers.Conv2dLayer(p2_1,
                               act = tf.identity,
                               shape = [5, 5, conv2_1,conv2_2],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer2_2')
    
    bn2 = tl.layers.BatchNormLayer(c2_2,is_train=True, act=tf.nn.relu, name='bn2')
    p2 = tl.layers.PoolLayer(bn2,
                             ksize=[1, 7, 7, 1],
                             strides=[1, 7, 7, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer2_2',)
    
    c3_1 = tl.layers.Conv2dLayer(data_in,
                               act = tf.identity,
                               shape = [7, 7, 3, conv3_1],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer3_1')
    bn3_1 = tl.layers.BatchNormLayer(c3_1,is_train=True, act=tf.nn.relu, name='bn3_1')
    p3_1 = tl.layers.PoolLayer(bn3_1,
                             ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer3_1',)
    c3_2 = tl.layers.Conv2dLayer(p3_1,
                               act = tf.identity,
                               shape = [7, 7, conv3_1, conv3_2],
                               strides=[1, 1, 1, 1],
                               padding='SAME',
                               W_init=tf.truncated_normal_initializer(stddev=5e-2),
                               W_init_args={},
                               b_init = tf.constant_initializer(value=0.0),
                               b_init_args = {},
                               name ='conv_layer3_2')
    
    bn3 = tl.layers.BatchNormLayer(c3_2,is_train=True, act=tf.nn.relu, name='bn3')
    p3 = tl.layers.PoolLayer(bn3,
                             ksize=[1, 7, 7, 1],
                             strides=[1, 7, 7, 1],
                             padding='SAME',
                             pool = tf.nn.max_pool,
                             name ='pool_layer3_2',)
    
    fla1 = tl.layers.FlattenLayer(p1,name='flatten_layer1')
    drop1_1 = tl.layers.DropoutLayer(fla1, keep=drop_out_rate1, name='drop1_1')
    
    fla2 = tl.layers.FlattenLayer(p2,name='flatten_layer2')
    drop2_1 = tl.layers.DropoutLayer(fla2, keep=drop_out_rate1, name='drop2_1')
    
    fla3 = tl.layers.FlattenLayer(p3,name='flatten_layer3')
    drop3_1 = tl.layers.DropoutLayer(fla3, keep=drop_out_rate1, name='drop3_1')
    
    concat = tl.layers.ConcatLayer([drop1_1,drop2_1,drop3_1],1,name='concat_layer_all')
    
    f1= tl.layers.DenseLayer(concat,
                              n_units = dense1,
                              act = tf.nn.relu,
                              W_init = tf.truncated_normal_initializer(stddev=0.1),
                              b_init = tf.constant_initializer(value=0.0),
                              W_init_args = {},
                              b_init_args = {},
                              name ='dense_layer1',)
    drop2 = tl.layers.DropoutLayer(f1, keep=drop_out_rate2, name='drop2')

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
    sess.close()
   
    return validation_acc


def get_reward(args):
# Hyper Parameters   
    cross_validation_acc=0
    batch_size = 200
    
    X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32,32,3),plotable=False)
    
    y_train=convert_to_one_hot(y_train,10)
       
    x_train = X_train
    
    cross_validation_acc=0
       
    conv1_1,conv1_2,conv2_1,conv2_2,conv3_1,conv3_2,drop_out_rate1,dense1,drop_out_rate2 = args
    
    conv1_1=np.floor(conv1_1)
    conv1_2=np.floor(conv1_2)   
    conv2_1=np.floor(conv2_1)
    conv2_2=np.floor(conv2_2) 
    conv3_1=np.floor(conv3_1)
    conv3_2=np.floor(conv3_2) 
    drop_out_rate1=drop_out_rate1*0.01
    dense1=np.floor(dense1)
    drop_out_rate2=drop_out_rate2*0.01
    
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
        sess = tf.InteractiveSession()
            
        x = tf.placeholder(tf.float32, shape=[None, 32, 32,3], name='x')
        y_ = tf.placeholder(tf.float32, shape=[None,10], name='y_')
        
        #architecture
        data_in = tl.layers.InputLayer(x, name='input_layer')
        
        data_in = tl.layers.BatchNormLayer(data_in,is_train=True, act=tf.nn.relu, name='bn0')
        c1_1 = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [3, 3, 3, conv1_1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_1')
        bn1_1 = tl.layers.BatchNormLayer(c1_1,is_train=True, act=tf.nn.relu, name='bn1_1')
        p1_1 = tl.layers.PoolLayer(bn1_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_1',)
        c1_2 = tl.layers.Conv2dLayer(p1_1,
                                   act = tf.identity,
                                   shape = [3, 3, conv1_1,conv1_2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer1_2')
        bn1 = tl.layers.BatchNormLayer(c1_2,is_train=True, act=tf.nn.relu, name='bn1')
        p1 = tl.layers.PoolLayer(bn1,
                                 ksize=[1, 7, 7, 1],
                                 strides=[1, 7, 7, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer1_2',)
             
        c2_1 = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [5, 5, 3, conv2_1],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer2_1')
        bn2_1 = tl.layers.BatchNormLayer(c2_1,is_train=True, act=tf.nn.relu, name='bn2_1')
        p2_1 = tl.layers.PoolLayer(bn2_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer2_1',)
        c2_2 = tl.layers.Conv2dLayer(p2_1,
                                   act = tf.identity,
                                   shape = [5, 5, conv2_1,conv2_2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer2_2')
        
        bn2 = tl.layers.BatchNormLayer(c2_2,is_train=True, act=tf.nn.relu, name='bn2')
        p2 = tl.layers.PoolLayer(bn2,
                                 ksize=[1, 7, 7, 1],
                                 strides=[1, 7, 7, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer2_2',)
        
        c3_1 = tl.layers.Conv2dLayer(data_in,
                                   act = tf.identity,
                                   shape = [7, 7, 3, conv3_1],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer3_1')
        bn3_1 = tl.layers.BatchNormLayer(c3_1,is_train=True, act=tf.nn.relu, name='bn3_1')
        p3_1 = tl.layers.PoolLayer(bn3_1,
                                 ksize=[1, 2, 2, 1],
                                 strides=[1, 2, 2, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer3_1',)
        c3_2 = tl.layers.Conv2dLayer(p3_1,
                                   act = tf.identity,
                                   shape = [7, 7, conv3_1, conv3_2],
                                   strides=[1, 1, 1, 1],
                                   padding='SAME',
                                   W_init=tf.truncated_normal_initializer(stddev=5e-2),
                                   W_init_args={},
                                   b_init = tf.constant_initializer(value=0.0),
                                   b_init_args = {},
                                   name ='conv_layer3_2')
        
        bn3 = tl.layers.BatchNormLayer(c3_2,is_train=True, act=tf.nn.relu, name='bn3')
        p3 = tl.layers.PoolLayer(bn3,
                                 ksize=[1, 7, 7, 1],
                                 strides=[1, 7, 7, 1],
                                 padding='SAME',
                                 pool = tf.nn.max_pool,
                                 name ='pool_layer3_2',)
        
        fla1 = tl.layers.FlattenLayer(p1,name='flatten_layer1')
        drop1_1 = tl.layers.DropoutLayer(fla1, keep=drop_out_rate1, name='drop1_1')
        
        fla2 = tl.layers.FlattenLayer(p2,name='flatten_layer2')
        drop2_1 = tl.layers.DropoutLayer(fla2, keep=drop_out_rate1, name='drop2_1')
        
        fla3 = tl.layers.FlattenLayer(p3,name='flatten_layer3')
        drop3_1 = tl.layers.DropoutLayer(fla3, keep=drop_out_rate1, name='drop3_1')
        
        concat = tl.layers.ConcatLayer([drop1_1,drop2_1,drop3_1],1,name='concat_layer_all')
        
        f1= tl.layers.DenseLayer(concat,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        drop2 = tl.layers.DropoutLayer(f1, keep=drop_out_rate2, name='drop2')
    
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


best = fmin(get_reward, space, algo=tpe.suggest,max_evals=1)

select_hp=[np.floor(best['conv1_1']),np.floor(best['conv1_2']),np.floor(best['conv2_1']),np.floor(best['conv2_2']),
           np.floor(best['conv3_1']),np.floor(best['conv3_2']),best['drop_out_rate1'],np.floor(best['dense1']),best['drop_out_rate2']]

test_error=get_reward_test(x_train, y_train, x_final_test, y_final_test, select_hp)


np.savetxt('pop_reserve_tpe_sl_conv_cifar10.txt',select_hp,delimiter=',')    

np.save('test_error_tpe_sl_conv_cifar10.npy',test_error) 