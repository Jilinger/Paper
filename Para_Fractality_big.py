#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 19:44:31 2019

@author: jp
"""

#%% Module

import random
import numpy as np
from numpy import linalg as LA
import scipy
from scipy import spatial
import time
import os
import copy
import multiprocessing as mp


#%% predefined Functions

#functions -------------------------------------------------------------------------------------------

from init_atoms import *


#%% Hamiltonian+Berechnung

def calculation(atoms):
    number_atoms=len(atoms)
    
    #Distance + Hamiltonian-------------------------------------------------------------------------------------
    distance_matrix = np.zeros((number_atoms,number_atoms),dtype=np.float16)
    distance_matrix= scipy.spatial.distance.cdist(atoms, atoms, metric='euclidean')
    del atoms
    
    np.fill_diagonal(distance_matrix,1)
    H=np.divide(coupling_constant,np.power(distance_matrix,3), out=distance_matrix)
    np.fill_diagonal(H,0)

#Linearisierung---------------------------------------------------------------------------------------------
    eigenvalues=np.zeros(number_atoms,dtype=np.float16)
    eigenvectors=np.zeros((number_atoms,number_atoms),dtype=np.float16)
    eigenvalues, eigenvectors = LA.eigh(H)
    del H
    
# Berechnung-----------------------------------------------------------------------------------------------
    p_eigenstates = np.square(np.absolute(eigenvectors), out=eigenvectors)  
   
    ipr_2=np.sum(p_eigenstates**2,axis=0)
    
    ratio = np.zeros([number_atoms])
    for n in range(1,number_atoms-1):
        delta_n = eigenvalues[n+1]-eigenvalues[n]
        delta_n_1 = eigenvalues[n]-eigenvalues[n-1]
        ratio[n] = np.min([delta_n,delta_n_1])/np.max([delta_n,delta_n_1])
        
    ratio[0]=-1
    ratio[-1]=-1
    
    return ipr_2,ratio, eigenvalues


global coupling_constant
global density
global number_atoms
global r_b
global radius
coupling_constant= -2.72*10**9  #stärke der nachbarwechselwirkung
density= 0.45
number_atoms=30000
r_b = 2.5
radius=np.sqrt(number_atoms*r_b**2/density)


#%% Generating folders

#folder_save= "/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Fractality/q_2/density_"+str(density)
#if not os.path.exists(folder_save):
#    os.makedirs(folder_save)
    
#
#folder_save= "/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Fractality/q_2/density_"+str(density)+"/atoms_"+str(number_atoms)
#if not os.path.exists(folder_save):
#   os.makedirs(folder_save)


#%% Mean function 

def average(aaa):
    coupling_constant= -2.72*10**9  #stärke der nachbarwechselwirkung
    atoms_array= produce_atoms(number_atoms, radius, r_b,versuch_max=15000)                 
    return calculation(atoms_array)


start=time.time()
iteration=10
pool = mp.Pool(mp.cpu_count())
ipr_2_all, ratio_all,eigenvalues_all=zip(*pool.map(average,np.arange(0,iteration,1)))
pool.close()
gebraucht=time.time()-start

print(ipr_2_all)
#data = open("/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Fractality/q_2/density_"+str(density)+"/atoms_"+str(number_atoms)+"/frac_"+str(number_atoms)+".npy","wb")
#text="Fractality: density=" +str(np.round(density,3))+", number_atoms="+str(number_atoms)+", average iterations=" + str(iteration)+", gebrauchte Zeit=" +str(gebraucht)+", r_b=" +str(r_b)
#np.save(data,text)
#np.save(data,gebraucht)
#np.save(data,ipr_2_all)                       
#np.save(data,eigenvalues_all)  
#np.save(data,ratio_all)  
#data.close()