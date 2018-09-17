#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 21:24:02 2018

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

from sklearn.gaussian_process import GaussianProcessRegressor

from sklearn.gaussian_process.kernels import Matern, RBF

X_train, y_train, X_val, y_val, X_test, y_test = tl.files.load_mnist_dataset(shape=(-1, 28,28))

def convert_to_one_hot(y, C):
    return np.eye(C)[y.reshape(-1)]

y_train=convert_to_one_hot(y_train,10)
y_val=convert_to_one_hot(y_val,10)
y_test=convert_to_one_hot(y_test,10)

x_final_test=X_test
y_final_test=y_test



x_train = np.vstack((X_train, X_val))
y_train =np.vstack((y_train, y_val))


DNA_SIZE = 7            # DNA length
POP_SIZE = 10          # population size
CROSS_RATE = 0.8         # mating probability (DNA crossover)
MUTATION_RATE = 0.2    # mutation probability
N_GENERATIONS = 30
x_bound = np.array([[10, 10,30,200,30,50,30],[70,70,70,300,70,100,70]])       # x upper and lower bounds
n_val=1#####number of validation


def get_reward_test(x_train, y_train, x_test, y_test, hp_setting):
    
    
    test_error=0
    for it in range(n_val): 
     
        tf.reset_default_graph()
        batch_size = 50
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
   
    return 1-test_error/n_val


def get_reward(x_train, y_train, hp_setting):
# Hyper Parameters   
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
        cross_validation_acc+=validation_acc
        sess.close()
    cross_validation_acc/=n_val    
    return cross_validation_acc


def surragate_fitness(x,pop,fitness,w):
    
   
    
    gp=GaussianProcessRegressor(
            kernel=RBF(length_scale=np.shape(pop)[1],length_scale_bounds=(1e-5,2*np.shape(pop)[1])),
            n_restarts_optimizer=25,
            
            )
       
    X=pop
    Y=fitness
    
    gp.fit(X,Y)
    
    s_fitness=np.zeros((np.shape(x)[0],1))

        
    mean,st=gp.predict(x,return_std=True)

    for i in range(np.shape(x)[0]):        
        ymean=mean[i][0]
        yvari=st[0]
        
        yvari2=yvari**2
        
        UCB=ymean+w*yvari
        
        gama=(np.min(fitness)-ymean)/yvari
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
        w=w_candi[gs%5]    
        pop_selected,fitness_selected=es_generation(pop_reserve,fitness_reserve,x_bound)
        
        pop_mutate_es=es_mutation_generation(pop_selected,fitness_selected,n_eachnitch,x_bound, mutation_rate)
    
        fitness_es=surragate_fitness(pop_mutate_es,pop_reserve,fitness_reserve,w)
        ind = fitness_es.argmax(axis=0)####maxmize UCB
        pop_child=pop_mutate_es[int(ind),]
        child_fitness=[[get_reward(x_train, y_train, pop_child)]]
        pop_reserve=np.concatenate(([pop_child], pop_reserve), axis=0)
        fitness_reserve=np.concatenate((child_fitness, fitness_reserve), axis=0)
    print(pop_reserve,fitness_reserve)
    
    
np.savetxt('pop_reserve_lenet_UBC_mnist.txt',pop_reserve,delimiter=',')    
np.savetxt('fitness_reserve_lenet_UBC_mnist.txt',fitness_reserve,delimiter=',') 


select_hp=pop_reserve[np.argmax(fitness_reserve), :]
test_error=get_reward_test(x_train, y_train, x_final_test, y_final_test, select_hp)


np.save('select_hp_lenet_UBC_mnist.txt',select_hp) 
np.save('test_error_lenet_UBC_mnist.npy',test_error) 



