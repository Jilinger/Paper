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
import multiprocessing as mp

#%% predefined Functions

#functions -------------------------------------------------------------------------------------------

from init_atoms import *


#%% decide whether atoms_array is saved or not
  
def find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z):
    if number_atoms>1000 and density >0.5:
            try:
                if save_nr<speicher_atome[z]:
                    atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","rb")
                    atoms_array= np.load(atoms)
                    atoms.close()
                    print("geopend")
                else:
                    atoms_array = produce_atoms(number_atoms, radius, r_b)
                    atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","wb")
                    np.save(atoms,atoms_array)
                    atoms.close()
            except  FileNotFoundError:
                atoms_array = produce_atoms(number_atoms, radius, r_b)
        
    else:
        atoms_array = produce_atoms(number_atoms, radius, r_b)  

    return atoms_array



#%% Hamiltonian

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
    
    
#Linearisierung---------------------------------------------------------------------------------------------
    #start = time.time()
    global eigenvalues
    global eigenvectors
    
    eigenvalues=np.zeros(number_atoms)
    eigenvectors=np.zeros((number_atoms,number_atoms))
    eigenvalues, eigenvectors = LA.eigh(H)
  
    #print("Time for Linearisierung: ",time.time()-start)
    
    return 


#%% function for IPR 

def calc_ipr():

    p_eigenstates = np.square(np.absolute(eigenvectors))  
   
    ipr_2=np.sum(p_eigenstates**2,axis=0)
    
    ratio = np.zeros([number_atoms])
    for n in range(1,number_atoms-1):
        delta_n = eigenvalues[n+1]-eigenvalues[n]
        delta_n_1 = eigenvalues[n]-eigenvalues[n-1]
        ratio[n] = np.min([delta_n,delta_n_1])/np.max([delta_n,delta_n_1])
        
    ratio[0]=-1
    ratio[-1]=-1
    return ipr_2,ratio







#%% Eingabe
    
global number_atoms
global radius
global r_b
coupling_constant= -2.72*10**9  #stärke der nachbarwechselwirkung
r_b = 2.5

#Parameter:
number_atoms=1000
densities= np.arange(0.01,0.51,0.01)

radiusse = (np.sqrt(number_atoms*r_b**2/densities))

iteration=5000


#%% Generating folders and counting configurations

folder_save= "/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Eigenstates/nr_atoms_"+str(number_atoms)
if not os.path.exists(folder_save):
    os.makedirs(folder_save)


#%% Mean function 

def average(density):  
   
    radius = (np.sqrt(number_atoms*r_b**2/density))
    
    start2=time.time()
    #folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)
    #save_nr = 0
    #atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z)                 
    
    atoms_array=produce_atoms(number_atoms, radius, r_b) 
    generate_hamiltonian(atoms_array)
    
    ipr_2_all,ratio_all = calc_ipr()
    eigenvalues_all= eigenvalues
    

    for i in range(0,iteration-1):
        #atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z) 
        atoms_array=produce_atoms(number_atoms, radius, r_b) 
        generate_hamiltonian(atoms_array)
       
        ipr_2_tmp, ratio_tmp = calc_ipr()
        
        ratio_all                       =  np.concatenate((ratio_all,ratio_tmp))
        eigenvalues_all                 =  np.concatenate((eigenvalues_all,eigenvalues))
        ipr_2_all                       =  np.concatenate((ipr_2_all,ipr_2_tmp))

    gebraucht = time.time()-start2
    
    data = open("/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Eigenstates/nr_atoms_"+str(number_atoms)+"/dens_"+str(np.round(density,3))+".npy","wb")
    text="Eigenstates,IPR+Ratios: density=" +str(np.round(density,3))+", number_atoms="+str(number_atoms)+", average iterations=" + str(iteration)+", gebrauchte Zeit=" +str(gebraucht)+", r_b=" +str(r_b)
    np.save(data,text)
    np.save(data,gebraucht)
    np.save(data,ipr_2_all)  
    np.save(data,ratio_all)                     
    np.save(data,eigenvalues_all)  
  
    data.close()
    
    
start1=time.time()
print(mp.cpu_count())
pool = mp.Pool(mp.cpu_count())

pool.map(average,densities)

pool.close()    

print("gesamt: ", time.time()-start1)


