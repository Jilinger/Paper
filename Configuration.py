# -*- coding: utf-8 -*-

# max density: 0.5472±0.002

#%% Module
import random
import numpy as np
import scipy
import time
import os
from scipy import spatial

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

#%% Eingabe und Ordner erzeugen


r_b=5
density=0.54


anzahl=np.array([2500,5000,7500,10000,12500,15000])
iteration=np.array([100,80,50,30,20,10])

anzahl=np.array([2500,5000])
iteration=np.array([10,10])



speicher_atome=np.zeros(len(anzahl),int)

s=0
for number_atoms in anzahl:
    radius = (np.sqrt(number_atoms*r_b**2/density))
    folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)
   
    if number_atoms>1000:
        if not os.path.exists(folder_a):
            os.makedirs(folder_a)
            print(folder_a +" erstellt")

        path, dirs, files = next(os.walk(folder_a))
        speicher_atome[s] = len(files)
    
    print("# files in Ordner atom_"+str(number_atoms)+": ", speicher_atome[s])
    s+=1



#%% Atome erzeugen

start=time.time()

z=0
ttt= np.zeros(len(anzahl[anzahl>500]))
for number_atoms in anzahl[anzahl>500]: 
    print("Atome: ", number_atoms)
    start2=time.time()
    for save_nr in range(speicher_atome[anzahl>500][z],iteration[anzahl>500][z]):
        print(save_nr)
        radius = (np.sqrt(number_atoms*r_b**2/density))
        atoms_array = produce_atoms(number_atoms, radius, r_b)
        folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)
        atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","wb")
        np.save(atoms,atoms_array)
        atoms.close()
    
    timo = time.time()-start2
    print("time: ",  timo)
    ttt[z] = timo
    z+=1
    
print("Total Time: ", time.time()-start)




