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

import cma
from numpy import array, dot, isscalar
from sympy import *
from scipy.integrate import quad,dblquad,nquad

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32,32,3),plotable=False)



def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

y_train=convert_to_one_hot(y_train,10)

y_test=convert_to_one_hot(y_test,10)

x_final_test=X_test
y_final_test=y_test



x_train = X_train


DNA_SIZE = 11            # DNA length
POP_SIZE = 2           # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.2    # mutation probability
N_GENERATIONS = 2
x_bound = np.array([[10, 10,10, 10,10,10,30,30,30,200,30],[70,70,70,70,70,70,70,70,70,400,70]])      # x upper and lower bounds
n_val=1#####number of validation




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
    drop_out_rate2=hp_setting[7]*0.01
    drop_out_rate3=hp_setting[8]*0.01
    dense1=hp_setting[9]
    drop_out_rate4=hp_setting[10]*0.01
    
            
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
    drop2_1 = tl.layers.DropoutLayer(fla2, keep=drop_out_rate2, name='drop2_1')
    
    fla3 = tl.layers.FlattenLayer(p3,name='flatten_layer3')
    drop3_1 = tl.layers.DropoutLayer(fla3, keep=drop_out_rate3, name='drop3_1')
    
    concat = tl.layers.ConcatLayer([drop1_1,drop2_1,drop3_1],1,name='concat_layer_all')
    
    f1= tl.layers.DenseLayer(concat,
                              n_units = dense1,
                              act = tf.nn.relu,
                              W_init = tf.truncated_normal_initializer(stddev=0.1),
                              b_init = tf.constant_initializer(value=0.0),
                              W_init_args = {},
                              b_init_args = {},
                              name ='dense_layer1',)
    drop2 = tl.layers.DropoutLayer(f1, keep=drop_out_rate4, name='drop2')

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


def get_reward(x_train, y_train, hp_setting):
# Hyper Parameters   
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
                          
        conv1_1=hp_setting[0]
        conv1_2=hp_setting[1]
        conv2_1=hp_setting[2]
        conv2_2=hp_setting[3]    
        conv3_1=hp_setting[4]
        conv3_2=hp_setting[5]        
        drop_out_rate1=hp_setting[6]*0.01
        drop_out_rate2=hp_setting[7]*0.01
        drop_out_rate3=hp_setting[8]*0.01
        dense1=hp_setting[9]
        drop_out_rate4=hp_setting[10]*0.01
        
                
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
        drop2_1 = tl.layers.DropoutLayer(fla2, keep=drop_out_rate2, name='drop2_1')
        
        fla3 = tl.layers.FlattenLayer(p3,name='flatten_layer3')
        drop3_1 = tl.layers.DropoutLayer(fla3, keep=drop_out_rate3, name='drop3_1')
        
        concat = tl.layers.ConcatLayer([drop1_1,drop2_1,drop3_1],1,name='concat_layer_all')
        
        f1= tl.layers.DenseLayer(concat,
                                  n_units = dense1,
                                  act = tf.nn.relu,
                                  W_init = tf.truncated_normal_initializer(stddev=0.1),
                                  b_init = tf.constant_initializer(value=0.0),
                                  W_init_args = {},
                                  b_init_args = {},
                                  name ='dense_layer1',)
        drop2 = tl.layers.DropoutLayer(f1, keep=drop_out_rate4, name='drop2')
    
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
    return cross_validation_acc


class SFitnessFunctions:  # TODO: this class is not necessary anymore? But some effort is needed to change it
    def __init__(self,pop,fitness):
        self.pop=pop
        self.fitness=fitness
    
    def gaussian(self, x):
        """Rosenbrock test objective function"""
        observe_x=self.pop
        observe_y=self.fitness
        K=np.zeros([observe_x.shape[0],observe_x.shape[0]])
        for i in range(observe_x.shape[0]):
            for j in range(observe_x.shape[0]):
                theta_d=x[2:]
                K[i,j]=x[1]*np.exp(np.sum(-(observe_x[i,]-observe_x[j,])**2*(1/(2*theta_d**2))))
        KI=K-(x[0]**2)*np.eye(observe_x.shape[0])
                     
        l1=-0.5*np.log(np.linalg.det(KI))
        
        l2=-0.5*np.dot(np.dot(observe_y.T,np.linalg.inv(KI)),observe_y)
        l2=l2[0,0]
        
        l3=-0.5*observe_x.shape[0]*np.log(2*np.pi)      
      
        return l1+l2+l3
        



def surragate(pop,fitness):  
    sff = SFitnessFunctions(pop,fitness)
    
    
    es=cma.CMAEvolutionStrategy((pop.shape[1]+2)*[0],0.5)

    es.optimize(sff.gaussian,iterations=200)

    para=es.result[0]#####optimal parameters
    return para



def surragate_fitness(x,para,pop,fitness,w):
    
    observe_x=pop
    observe_y=fitness
    
    K=np.zeros([pop.shape[0],pop.shape[0]])
    s_fitness=np.zeros((np.shape(x)[0],1))
    
    for i in range(pop.shape[0]):
        for j in range(pop.shape[0]):
            theta_d=para[2:]
            K[i,j]=para[1]*np.exp(np.sum(-(pop[i,]-pop[j,])**2*(1/(2*theta_d**2))))
    KI=K-(para[0]**2)*np.eye(pop.shape[0])
        
    for i in range(np.shape(x)[0]):
        test_x=x[i,]
        
        Kx=np.zeros([1,pop.shape[0]])
        for j in range(pop.shape[0]):
            theta_d=para[2:]
            Kx[0,j]=para[1]*np.exp(np.sum(-(test_x-pop[j,])**2*(1/(2*theta_d**2))))
        ymean=np.dot(np.dot(Kx,np.linalg.det(KI)),fitness)[0,0]
        yvari2=para[1]-np.dot(np.dot(Kx,np.linalg.det(KI)),Kx.T)[0,0]
        yvari=yvari2**(-0.5)
        
        UCB=ymean+w*yvari
        
        gama=(np.min(observe_y)-ymean)/yvari
        PI=(2*np.pi*yvari2)**(-0.5)*quad(lambda x:np.exp(-(x-ymean)**2/(2*yvari2)),-np.inf,gama)[0]
        EI=yvari*gama*PI+yvari*(2*np.pi*yvari2)**(-0.5)*np.exp(-gama)
                       
        s_fitness[i,0]=EI
        
    return s_fitness


def es_generation(pop_reserve,fitness_reserve,x_bound):#####get two solution in every dimension
    n=np.shape(pop_reserve)[0]
    d=np.shape(pop_reserve)[1]
    pop_selected=pop_reserve[0,]
    fitness_selected=fitness_reserve[0,]    
    
    for i in range(d):
        pp=pop_reserve[:,[i]]
        index=np.mat(np.arange(n)).T
        p =np.hstack((pp, index))
        np.random.shuffle(p)
        pop1=pop_reserve[int(p[0, 1]),]
        fitness1=fitness_reserve[int(p[0, 1]),]
        pop2=pop_reserve[int(p[1, 1]),]
        fitness2=fitness_reserve[int(p[1, 1]),]
        for j in range(2,n):
            if p[j,0]<0.5*(x_bound[1,i]-x_bound[0,i])+x_bound[0,i]:
               if fitness_reserve[int(p[j, 1]),]>fitness1:
                  pop1=pop_reserve[int(p[j, 1]),]
                  fitness1=fitness_reserve[int(p[j, 1]),]
            elif p[j,0]>=0.5*(x_bound[1,i]-x_bound[0,i])+x_bound[0,i]:
                 if fitness_reserve[int(p[j, 1]),]>fitness2:
                    pop2=pop_reserve[int(p[j, 1]),]
                    fitness2=fitness_reserve[int(p[j, 1]),]
        pop12=np.vstack((pop1, pop2))
        fitness12=np.vstack((fitness1, fitness2))
        pop_selected=np.vstack((pop_selected, pop12))
        fitness_selected=np.vstack((fitness_selected, fitness12))
    return pop_selected,fitness_selected


def es_mutation_generation(pop_selected,fitness_selected,n_eachnitch,x_bound, mutation_rate):####mutate every dimensional solutions
    n=np.shape(pop_selected)[0]
    d=np.shape(pop_selected)[1]
    
    ind = fitness_selected.argmax(axis=0)
    mutate_pop=pop_selected[ind,]
    for i in range(n):
        pop_mutate=pop_selected[i,]
        child=np.zeros((n_eachnitch,d))
        for j in range(n_eachnitch):
            child[j,]=pop_mutate
            for point in range(d):
                if np.random.rand() < mutation_rate:
                   child[j,point]=np.random.randint(x_bound[0,point],x_bound[1,point])
        mutate_pop=np.vstack((mutate_pop, child))
    return mutate_pop


def hyper_initial_pop(x_bound,n_point):
    x_min=x_bound[0,]
    x_max=x_bound[1,]
    lhd = lhs(len(x_min),samples=n_point) 
    mm=x_max-x_min
    lhd=np.transpose(lhd)
    mm=x_max-x_min
    ld=np.zeros((len(x_min),n_point))
    for i in range(len(x_min)):
        ld[i,]=x_min[i]+lhd[i,]*mm[i]
    ld=np.around(ld)
    ld=np.transpose(ld)
    return ld

pop = hyper_initial_pop(x_bound,POP_SIZE)

print(pop.shape)

ss=np.shape(pop)[0]

fitness=np.zeros((ss,1))
for i in range(ss):
    fitness[i,]=[get_reward(x_train, y_train, pop[i,])]  

pop_reserve=pop
fitness_reserve=fitness
mutation_rate=0.2
n_eachnitch=100
w_candi=[0.1,0.2,0.4,0.8,0.95]
for gs in range(N_GENERATIONS):  

    print("Most fitted DNA: ", pop_reserve[np.argmax(fitness_reserve), :])
    print("Most fitted DNA: ", max(fitness_reserve))
    for oo in range(1):
#    oo=_%4
        w=w_candi[oo]    
        s_p=surragate(pop_reserve,fitness_reserve)
        pop_selected,fitness_selected=es_generation(pop_reserve,fitness_reserve,x_bound)
        
        pop_mutate_es=es_mutation_generation(pop_selected,fitness_selected,n_eachnitch,x_bound, mutation_rate)
        
    
        fitness_es=surragate_fitness(pop_mutate_es,s_p,pop_reserve,fitness_reserve,w)
        ind = fitness_es.argmax(axis=0)
        pop_child=pop_mutate_es[int(ind),]
        child_fitness=[[get_reward(x_train, y_train, pop_child)]]
        pop_reserve=np.concatenate(([pop_child], pop_reserve), axis=0)
        fitness_reserve=np.concatenate((child_fitness, fitness_reserve), axis=0)
    print(pop_reserve,fitness_reserve)

np.savetxt('pop_reserve_sl_mnist.txt',pop_reserve,delimiter=',')    
np.savetxt('fitness_reserve_sl_mnist.txt',fitness_reserve,delimiter=',') 


select_hp=pop_reserve[np.argmax(fitness_reserve), :]
test_error=get_reward_test(x_train, y_train, x_final_test, y_final_test, select_hp)

np.save('test_error_sl_cifar10.npy',test_error) 

