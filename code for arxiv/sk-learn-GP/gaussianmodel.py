#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 17:16:57 2018

@author: mzhang3
"""

import cma
import numpy as np
from numpy import array, dot, isscalar
from sympy import *
from scipy.integrate import quad,dblquad,nquad



kk=1e2



class SFitnessFunctions(object):  # TODO: this class is not necessary anymore? But some effort is needed to change it
   
    
    def gaussian(self, x,observe_x=observe_x,observe_y=observe_y):
        """Rosenbrock test objective function"""
        
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
        
sff = SFitnessFunctions()
observe_x=np.random.random([10,5])
observe_y=np.ones([10,1])*[[1,2,3,4,5]]


es=cma.CMAEvolutionStrategy(7*[0],0.5)

es.optimize(sff.gaussian,iterations=200)

para=es.result[0]#####optimal parameters


K=np.zeros([observe_x.shape[0],observe_x.shape[0]])
for i in range(observe_x.shape[0]):
    for j in range(observe_x.shape[0]):
        theta_d=para[2:]
        K[i,j]=para[1]*np.exp(np.sum(-(observe_x[i,]-observe_x[j,])**2*(1/(2*theta_d**2))))
KI=K-(para[0]**2)*np.eye(observe_x.shape[0])



test_x=np.random.random([1,5])######### test point
Kx=np.zeros([1,observe_x.shape[0]])
for i in range(observe_x.shape[0]):
    theta_d=para[2:]
    Kx[0,i]=para[1]*np.exp(np.sum(-(test_x-observe_x[i,])**2*(1/(2*theta_d**2))))
ymean=np.dot(np.dot(Kx,np.linalg.det(KI)),observe_y)[0,0]
yvari2=para[1]-np.dot(np.dot(Kx,np.linalg.det(KI)),Kx.T)[0,0]
yvari=yvari2**(-0.5)


###### calculate the UCB
w=0.5
UCB=ymean+w*yvari




###### calculate the PI
gama=(np.min(observe_y)-ymean)/yvari


PI=(2*np.pi*yvari2)**(-0.5)*quad(lambda x:np.exp(-(x-ymean)**2/(2*yvari2)),-np.inf,gama)[0]

EI=yvari*gama*PI+yvari*(2*np.pi*yvari2)**(-0.5)*np.exp(-gama)

#EI=gama*(2*np.pi*yvari2)**(-0.5)*quad(lambda x:np.exp(-(x-ymean)**2/(2*yvari2),-np.inf,gama)[0]+(2*np.pi)**(-0.5)*np.exp(-(x-ymean)**2/(2*yvari2))