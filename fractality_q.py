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


#%% predefined Functions

#functions -------------------------------------------------------------------------------------------

def to_ind(coord):
    x_shifted = coord[0]+radius
    y_shifted = np.abs(coord[1]-radius)

    j = np.floor(x_shifted/(size_square))
    i = np.floor(y_shifted/(size_square))
    return int(i),int(j)

def to_coord(ind):
    x = ind[1]*size_square
    y = ind[0]*size_square
    
    x_shifted = x -radius
    y_shifted = radius - y
 
    return [x_shifted, y_shifted]


def get_atoms_near(punkt, counting_atoms):
    m,n = to_ind(punkt[0])
    atoms_near=[]
    atoms_near.extend(counting_atoms[m][n][1])

    try:
        atoms_near.extend(counting_atoms[m+1][n][1])             
        atoms_near.extend(counting_atoms[m+1][n+1][1])       
        atoms_near.extend(counting_atoms[m][n+1][1])
        atoms_near.extend(counting_atoms[m-1][n-1][1])        
        atoms_near.extend(counting_atoms[m-1][n][1])
        atoms_near.extend(counting_atoms[m-1][n+1][1])
        atoms_near.extend(counting_atoms[m][n-1][1])
        atoms_near.extend(counting_atoms[m+1][n-1][1])
        return atoms_near
    except IndexError:
        if m+1<number_squares:
            atoms_near.extend(counting_atoms[m+1][n][1])
            if n+1< number_squares:
                atoms_near.extend(counting_atoms[m+1][n+1][1])
        if n+1<number_squares:
            atoms_near.extend(counting_atoms[m][n+1][1])

        if (m-1)>=0 and (n-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n-1][1])
        if (m-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n][1])
            if n+1<number_squares:
                atoms_near.extend(counting_atoms[m-1][n+1][1])
        if (n-1)>=0: 
            atoms_near.extend(counting_atoms[m][n-1][1])
            if m+1<number_squares:
                atoms_near.extend(counting_atoms[m+1][n-1][1])
    return atoms_near


#produce atoms ohne Vorschlagen bei letzten Atomen wohin es plaziert werden soll

def produce_atoms(number_atoms, a, r_b):
    global size_square
    global number_squares
    size_square = 2*r_b #np.sqrt(density*50*r_b**2)   #2*r_b   #wähle size_square so, dass 50 atome drin sind.
    #size_square gerade klein genug um 8 Felder außen rum zu benutzen um abzugleichen
    number_squares= int(a*2/size_square)
    size_square = a*2/number_squares
    

    counting_atoms = [[[0,[]] for i in range(number_squares)] for i in range(number_squares)]
        
    atoms = [[0,0]]
    m,n = to_ind([0,0])
    counting_atoms[m][n][0]+=1
    counting_atoms[m][n][1] = [[0,0]]
    
    #start=time.time()
    for i in range(number_atoms-len(atoms)):
        r=(a-r_b)*np.sqrt(random.uniform(0,1))
        theta=random.uniform(0,1)*2*np.pi
        x,y=r * np.cos(theta), r * np.sin(theta)
        punkt = np.array([[x,y]]) 
        gesetzt= False
        
        while (gesetzt==False):
            gesetzt=True

            atoms_near = get_atoms_near(punkt, counting_atoms)

            if atoms_near:
                if ((np.min(scipy.spatial.distance.cdist(atoms_near, punkt, metric='euclidean')))<(2*r_b)):
                    r=a*np.sqrt(random.uniform(0,1))
                    theta=random.uniform(0,1)*2*np.pi
                    x,y=r * np.cos(theta), r * np.sin(theta)
                    punkt = np.array([[x,y]]) 
                    gesetzt=False
                            
                    
        atoms.append(punkt[0])
        m,n = to_ind(punkt[0])
        counting_atoms[m][n][0]+=1
        counting_atoms[m][n][1].append(punkt[0].tolist())

    return np.round(atoms,4)

#%% decide whether atoms_array is saved or not
  
def find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z):
    if number_atoms>1000 and density >0.5:
            try:
                if save_nr<speicher_atome[z]:
                    atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","rb")
                    atoms_array= np.load(atoms)
                    atoms.close()
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
    
    eigenvalues=np.zeros(number_atoms,dtype=np.float16)
    eigenvectors=np.zeros((number_atoms,number_atoms),dtype=np.float16)
    
    eigenvalues, eigenvectors = LA.eigh(H)
  
    #print("Time for Linearisierung: ",time.time()-start)
    
    return 


#%% function for IPR 


def calc_ipr_q(q_array):

    p_eigenstates = np.square(np.absolute(eigenvectors))  
   
    #inverse participation ratio ------------------------------------------------------------------------
 
    ipr_2=np.sum(p_eigenstates**2,axis=0)
    
    ipr_q = np.zeros([len(q_array),number_atoms])
    
    for i,q in enumerate(q_array):
            ipr_q[i]= np.sum(p_eigenstates**q,axis=0)
        
    return ipr_2, ipr_q







#%% Eingabe
    
global number_atoms
global radius
global r_b
coupling_constant= 3*10**9  #stärke der nachbarwechselwirkung
r_b = 5


#Parameter:
density= 0.54


anzahl = np.array([100,500,1000,1500,2000,2500,5000,7500,10000,12500,15000])
radiusse = (np.sqrt(anzahl*r_b**2/density))
#iteration= [500,400,200,60,60,50,50,40,25,25,20]




#%% Generating folders and counting configurations

speicher_atome=np.zeros(len(anzahl),int)

s=0
for number_atoms, radius in zip(anzahl,radiusse): 
    folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)   

    if number_atoms>1000:
        if not os.path.exists(folder_a):
            os.makedirs(folder_a)
            print(folder_a +" erstellt")

        path, dirs, files = next(os.walk(folder_a))
        speicher_atome[s] = len(files)
    
    print("# files in Ordner atom_"+str(number_atoms)+": ", speicher_atome[s])
    s+=1

iteration= speicher_atome
iteration[0] = 500
iteration[1] = 200
iteration[2] = 100


q_array = np.arange(2,30,1)


#%% Mean function 
save_nr=0
z=0

start1=time.time()
gebrauchte_zeit=np.zeros(len(anzahl))

for number_atoms, radius in zip(anzahl,radiusse):  
    print(number_atoms)
    print()
    
    start2=time.time()
    folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)
    save_nr = 0
        
    atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z)                 
    generate_hamiltonian(atoms_array)
    
    ipr_2_all,ipr_q_all = calc_ipr_q(q_array)
    eigenvalues_all= eigenvalues
    

    save_nr+=1
    for i in range(0,iteration[z]-1):
        
        atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr,z)        
        generate_hamiltonian(atoms_array)
       
        ipr_2_tmp, ipr_q_tmp = calc_ipr_q(q_array)
        
        #ratio_all                      =  np.concatenate((ratio_all,ratio_tmp))
        eigenvalues_all                 =  np.concatenate((eigenvalues_all,eigenvalues))
        ipr_2_all                       =  np.concatenate((ipr_2_all,ipr_2_tmp))
        ipr_q_all                       =  np.concatenate((ipr_q_all,ipr_q_tmp), axis=1)
    
        save_nr+=1
        
    gebrauchte_zeit[z]=(time.time()-start2)
    print(gebrauchte_zeit[z])
    z+=1


    data = open("/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Fractality/different_q/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)+".npy","wb")
    
    np.save(data,ipr_2_all)   
    #np.save(data,ratio_all)                       
    np.save(data,eigenvalues_all)  
    np.save(data,ipr_q_all)  
  
   
    data.close()
    
print(gebrauchte_zeit/60/60)
print(time.time()-start1)


