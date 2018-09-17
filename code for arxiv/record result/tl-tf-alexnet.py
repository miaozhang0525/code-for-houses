#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 16:40:48 2018

@author: jiahzhao
"""
import os
import time
import tensorflow as tf
import tensorlayer as tl
import tflearn

import numpy as np



tf.reset_default_graph()

X_train, y_train, X_test, y_test = tl.files.load_cifar10_dataset(shape=(-1, 32,32,3),plotable=False)
# X_train, y_train, X_test, y_test = tl.files.load_cropped_svhn(include_extra=False)

y_train = tflearn.data_utils.to_categorical(y_train,10)

y_test = tflearn.data_utils.to_categorical(y_test,10)

rootdir='/home/jiahzhao/Desktop/result for houses-ALL-final/alexnet-result'

result_list=os.listdir(rootdir)

for ii in range(0,len(result_list)):
    
    tf.reset_default_graph()

    path=os.path.join(rootdir,result_list[ii])
    result_name=os.path.basename(path)
    
    print(result_name)
    if result_name[-3:]=='txt':
    
       hp_setting=np.loadtxt(path,delimiter=',')
    elif result_name[-3:]=='npy':
       hp_setting=np.load(path)
    
    hp_setting=np.round(hp_setting)
    conv1=hp_setting[0]
    conv2=hp_setting[1]
    conv3=hp_setting[2]
    conv4=hp_setting[3]
    conv5=hp_setting[4]
    dense1=hp_setting[5]
    drop_out_rate1=hp_setting[6]*0.01
    dense2=hp_setting[7]
    drop_out_rate2=hp_setting[8]*0.01
    
    
    sess = tf.InteractiveSession()
    
    batch_size = 50
    
    x = tf.placeholder(tf.float32, shape=[batch_size, 32,32,3])
    y_ = tf.placeholder(tf.float32, shape=[batch_size,10])
    
    
    def model(x, is_train=True, reuse=False):
        # In BNN, all the layers inputs are binary, with the exception of the first layer.
        # ref: https://github.com/itayhubara/BinaryNet.tf/blob/master/models/BNN_cifar10.py
        with tf.variable_scope("binarynet", reuse=reuse):
    
            
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
            drop1= tl.layers.DropoutLayer(f1, keep=drop_out_rate1,is_fix=True, is_train=is_train,  name='drop1')
            f2= tl.layers.DenseLayer(drop1,
                                      n_units = dense2,
                                      act = tf.nn.relu,
                                      W_init = tf.truncated_normal_initializer(stddev=0.1),
                                      b_init = tf.constant_initializer(value=0.0),
                                      W_init_args = {},
                                      b_init_args = {},
                                      name ='dense_layer2',)
            drop2= tl.layers.DropoutLayer(f2, keep=drop_out_rate2, is_fix=True, is_train=is_train, name='drop2')    
            net = tl.layers.DenseLayer(drop2, n_units=10,
                                             act = tf.identity,
                                             name='output_layer')
    
            
            
          
        return net
    
    
    # define inferences
    net_train = model(x, is_train=True, reuse=False)
    net_test = model(x, is_train=False, reuse=True)
    
    # cost for training
    y = net_train.outputs
    cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=y_), name='xentropy')
    
    
    # cost and accuracy for evalution
    y2 = net_test.outputs
    cost_test = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y2, labels=y_), name='xentropy2')
    
    correct_prediction = tf.equal(tf.argmax(y2,1), tf.argmax(y_,1))
    
    
    
    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    # define the optimizer
    train_params = tl.layers.get_variables_with_name('binarynet', True, True)
    train_op = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(cost, var_list=train_params)
    
    # initialize all variables in the session
    tl.layers.initialize_global_variables(sess)
    
    net_train.print_params()
    net_train.print_layers()
    
    n_epoch = 200
    
    
    # print(sess.run(net_test.all_params)) # print real values of parameters
    record=np.zeros((1,5))#####save a  set with n(epoches+1)row 5 clomn(epoch,trainloss trainacc,testloss,testacc)
    
    
    for epoch in range(1,n_epoch+1):
        start_time = time.time()
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            sess.run(train_op, feed_dict={x: X_train_a, y_: y_train_a})
    
        print("Epoch %d of %d took %fs" % (epoch, n_epoch, time.time() - start_time))
        train_loss, train_acc, n_batch = 0, 0, 0
        for X_train_a, y_train_a in tl.iterate.minibatches(X_train, y_train, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_train_a, y_: y_train_a})
            train_loss += err
            train_acc += ac
            n_batch += 1
    
        print("   train loss: %f" % (train_loss / n_batch))
        print("   train acc: %f" % (train_acc / n_batch))
        
        train_loss=  train_loss/n_batch                  
        train_acc=train_acc/n_batch    
        
        val_loss, val_acc, n_batch = 0, 0, 0
        for X_val_a, y_val_a in tl.iterate.minibatches(X_test, y_test, batch_size, shuffle=True):
            err, ac = sess.run([cost_test, acc], feed_dict={x: X_val_a, y_: y_val_a})
            val_loss += err
            val_acc += ac
            n_batch += 1
        print("   val loss: %f" % (val_loss / n_batch))
        print("   val acc: %f" % (val_acc / n_batch))
        val_loss= val_loss / n_batch
        val_acc=val_acc / n_batch           
        
        record_e=[[epoch,train_loss,train_acc,val_loss,val_acc]]
        
        print(record_e)
        print(record)
        record=np.concatenate((record,record_e),axis=0)
        print(record)
    record=record[1:,] 
    print('epoch,train_loss,train_acc,val_loss,val_acc') 
    print(record)

    sname='saved_result_'+result_name+'.txt'

    np.savetxt(sname,record)