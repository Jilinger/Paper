#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 20:08:43 2020

@author: jp
"""


import math
import numpy as np
import scipy.spatial
from numpy import linalg as LA
import copy
import time
import gc

# In[19]:
from init_atoms import *



def generate_hamiltonian(atoms):
    number_atoms=len(atoms)
    
    #Distance + Hamiltonian-------------------------------------------------------------------------------------
    H = np.zeros((number_atoms,number_atoms),dtype=np.float16)
    global distance_matrix
    distance_matrix = np.zeros((number_atoms,number_atoms),dtype=np.float16)
    
    
    #start= time.time()
    distance_matrix= scipy.spatial.distance.cdist(atoms, atoms, metric='euclidean')
    np.fill_diagonal(distance_matrix,1)
    H=np.divide(coupling_constant,np.power(distance_matrix,3))
    np.fill_diagonal(distance_matrix,0)
    np.fill_diagonal(H,0)
    #print("Time for Hamiltonian: ",time.time()-start)
    
    
    global abstand_ursprung
    abstand_ursprung= np.zeros(number_atoms, dtype=np.float16)
    abstand_ursprung = copy.copy(distance_matrix[:,angeregt])
    
#Linearisierung---------------------------------------------------------------------------------------------
    #start = time.time()
    global eigenvalues
    eigenvalues=np.zeros(number_atoms,dtype=np.float16)
    global eigenvectors2
    eigenvectors=np.zeros((number_atoms,number_atoms),dtype=np.float16)
    eigenvalues, eigenvectors = LA.eigh(H)
    eigenvectors2=copy.copy(eigenvectors)
    #print("Time for Linearisierung: ",time.time()-start)

    return eigenvalues, eigenvectors,H

#------------------------------------------------------------------------------------------------

global coupling_constant
global gamma
global number_atoms
global atoms_array
global H
global angeregt

coupling_constant= -2.72*10**9 
r_b=2.5   
angeregt=0
iteration=1
#atoms=np.arange(1000,15500,500)

atoms=np.array([40000])
times_dens_05=[[[[],[]] for i in range(iteration)] for i in range(len(atoms))]
times_dens_01=[[[[],[]] for i in range(iteration)] for i in range(len(atoms))]


both_densities=False
for k,number_atoms in enumerate(atoms):
    print(number_atoms)
    density=0.5
    radius = (np.sqrt(number_atoms*r_b**2/density))
    for j in range(iteration):
        start=time.time()
        atoms_array=produce_atoms(number_atoms, radius, r_b) 
        time1=time.time()-start
        start=time.time()
        eigenvalues, eigenvectors,H = generate_hamiltonian(atoms_array)
        time2=time.time()-start
        times_dens_05[k][j][0]=time1
        times_dens_05[k][j][1]=time2
        
    if both_densities:
        density=0.1
        radius = (np.sqrt(number_atoms*r_b**2/density))
        for j in range(iteration):
            start=time.time()
            atoms_array=produce_atoms(number_atoms, radius, r_b) 
            time1=time.time()-start
            start=time.time()
            eigenvalues, eigenvectors,H = generate_hamiltonian(atoms_array)
            time2=time.time()-start
            times_dens_01[k][j][0]=time1
            times_dens_01[k][j][1]=time2

data = open("times_for_diagonalizing_big4.npy","wb")
np.save(data,times_dens_05)
np.save(data,times_dens_01)
    
