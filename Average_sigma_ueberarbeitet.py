#!/usr/bin/env python
# coding: utf-8

# In[3]:


import math
import random
#Benötigte Pakete


import numpy as np


from numpy import linalg as LA
import scipy
from scipy import spatial
import time
#import tables

import os
#import h5py
import copy


# In[4]:


def distance(x1,y1,x2,y2):
    dx=x1-x2
    dy=y1-y2
    return math.hypot(dx, dy)

def e_func (x,L,a):
    return a*np.exp(-x/L)
coords=[];


# ### Eingabe


# In[6]:


def quad_func(t,a):
    return a*t**2

def lin_func(t,a):
    return a*t

def sublin_func(t,a,b):
    return a*t**b


# ### Erzeugung

# In[7]:


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
    

def find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr):
    if number_atoms>1000 and density >=0.5:
            try:
                if save_nr<speicher_atome:
                    atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","rb")
                    atoms_array= np.load(atoms)
                    atoms.close()
                else:
                    atoms = open(folder_a+"/atoms"+"_"+str(save_nr)+".npy","wb")
                    atoms_array = produce_atoms(number_atoms, radius, r_b)
                    np.save(atoms,atoms_array)
                    atoms.close()
            except  FileNotFoundError:
                atoms_array = produce_atoms(number_atoms, radius, r_b)
        
    else:
        atoms_array = produce_atoms(number_atoms, radius, r_b)  

    return atoms_array


# ### Wahrscheinlichkeitsberechnung

# In[8]:


def generate_hamiltonian(atoms):
    number_atoms=len(atoms)
    
    #Distance + Hamiltonian-------------------------------------------------------------------------------------
    H = np.zeros((number_atoms,number_atoms))
    global distance_matrix
    distance_matrix = np.zeros((number_atoms,number_atoms))
    
    
    #start= time.time()
    distance_matrix= scipy.spatial.distance.cdist(atoms, atoms, metric='euclidean')
    np.fill_diagonal(distance_matrix,1)
    H=np.divide(coupling_constant,np.power(distance_matrix,3))
    np.fill_diagonal(distance_matrix,0)
    np.fill_diagonal(H,0)
    #print("Time for Hamiltonian: ",time.time()-start)
    
    
    global abstand_ursprung
    abstand_ursprung= np.zeros(number_atoms)
    abstand_ursprung = copy.copy(distance_matrix[:,angeregt])
    
#Linearisierung---------------------------------------------------------------------------------------------
    #start = time.time()
    global eigenvalues
    global eigenvectors
    eigenvalues=np.zeros(number_atoms)
    eigenvectors=np.zeros((number_atoms,number_atoms))
    eigenvalues, eigenvectors = LA.eigh(H)
    #print("Time for Linearisierung: ",time.time()-start)
    
    
    return eigenvalues, eigenvectors


# In[9]:


def ultimate_justeins(eigen_values,eigenvectors,times_tmp,angeregt):
    
    total_steps=len(times_tmp)
#Zeitenwicklung--------------------------------------------------------------------------------------------
    #start = time.time()
    projection=eigenvectors[angeregt]
    psi_t= np.zeros((total_steps,number_atoms,1),dtype="complex")
    for i,t in enumerate(times_tmp):
        psi_t[i]= (eigenvectors @ (np.exp(-1j*eigenvalues*t)*projection))[:,None]
        
    #print("Time for Probability_Zeitentwicklung: ",time.time()-start)
    
    #komplette Zeitenwicklung ohne Schleife, aber leider langsamer:
            #psi_t2= ((eigenvectors*np.exp(-1j*eigen_values*times_tmp[:,None])[:,None])@projection)[:,:,None]
    
    
#Wahrscheinlichkeiten---------------------------------------------------------------------------------------
    #start = time.time()
    np.square(np.absolute(psi_t,out=psi_t),out=psi_t)    # now psi_all_t = probability
    #print("Time for Probability_wahrscheinlichkeit: ",time.time()-start)

    probability=np.real(psi_t)
  
    return probability


# In[10]:


def Berechnung(eigenvalues,eigenvectors,number_atoms,time_array):  
    
    probability = np.zeros((1,number_atoms,1))
    anzahl_steps_möglich =  math.floor(1000000000/(number_atoms*number_atoms))
    # Restriction for each PC for how many time steps it can handel
    
    global runden
    ang_anzahl_steps=round(len(time_array))
    runden = math.floor((ang_anzahl_steps-1)/anzahl_steps_möglich)
    print("Von ",time_array[0],"s bis",time_array[-1],"s mit", ang_anzahl_steps," steps")
    print("Maximal mögliche Anzahl an Steps auf einmal:",anzahl_steps_möglich )
    print("Anzahl Wiederholungen:",runden)
    
    start=time.time()

    for i in range(runden):
        times_tmp = time_array[i*anzahl_steps_möglich:(i+1)*anzahl_steps_möglich]
        print("round ", i, "  steps: ",len(times_tmp), " from: ",times_tmp[0], " to ", times_tmp[-1])
        
        prob_tmp = ultimate_justeins(eigenvalues,eigenvectors,times_tmp,angeregt)
        probability=np.append(probability,prob_tmp,axis=0)
    
    times_tmp = time_array[runden*anzahl_steps_möglich:]
    print("round ", runden, "  steps: ",len(times_tmp), " from: ",times_tmp[0], " to ", times_tmp[-1])
    prob_tmp = ultimate_justeins(eigenvalues,eigenvectors,times_tmp,angeregt)
    probability=np.append(probability,prob_tmp,axis=0)
        
    print("Berechnung:", np.round(time.time()-start,3))
       
       
    probability=np.delete(probability,0,axis=0)
                         
    return probability


# In[11]:


def auswertung(probability):
    
    
    # Mean square und mean displacement---------------------------------------------------------------------
    r_2=np.dot(np.power(abstand_ursprung,2), probability)[:,0]
    r_1=np.dot(abstand_ursprung, probability)[:,0]
    
    
    #standard deviation -----------------------------------------------------------------------------------
    deviation=np.subtract(r_2,np.square(r_1))
    
    
    #infinity_time-------------------------------------------------------------------------------------
    p_infinity=np.dot(np.square(eigenvectors[angeregt]),np.square(eigenvectors.transpose()))

    
    ipr = []
    for i in range(len(probability)):
        ipr.append((1/(np.sum(probability[i][:,angeregt]**2))))
    
    p_infinity_g=np.dot(np.power(eigenvectors[angeregt],4),np.power(eigenvectors.transpose(),4))
    
    ipr_inf = 1/(np.sum(2*p_infinity**2-p_infinity_g))

    
    
    
    #radial densitiy--------------------------------------------------------------
      
    n_intervall_inf=[]
    density_inf=[] 
   
    global d_r
    d_r=r_b*2
    
    for i in range(0,int(radius-d_r),d_r):
        flaeche= np.pi*((i+d_r)**2-i**2)
    
        n_intervall_inf.append((np.sum(p_infinity[(i<=abstand_ursprung)& (abstand_ursprung<i+d_r)]))/(flaeche))
            
        index = np.argwhere((i<=abstand_ursprung)& (abstand_ursprung<i+d_r))
        laenge= len(index)
        if laenge==0:
            density_inf.append(0)
        else:
            density_inf.append(np.sum(p_infinity[index])/len(index))
    
        
    r_2_inf=np.dot(p_infinity,np.power(abstand_ursprung,2)[:,None])
    r_1_inf=np.dot(p_infinity,abstand_ursprung[:,None])
    deviation_inf=np.subtract(r_2_inf,np.square(r_1_inf))
    
    
    return np.array(r_1),np.array(r_2), np.array(r_1_inf), np.array(r_2_inf), np.array(deviation), np.array(deviation_inf), np.array(density), np.array(density_inf),np.array(ipr), np.array(ipr_inf), np.array(n_intervall_inf)


# ## MEAN

# #### Mittelung über Startanregungen

# In[20]:


global number_atoms
global radius
coupling_constant= 3*10**9  #stärke der nachbarwechselwirkung
global angeregt
angeregt=0                  #atom 0 is atom in the center

#Parameter:
number_atoms=10000
density= 0.2
r_b = 5
radius = (np.sqrt(number_atoms*r_b**2/density))


time_array=np.logspace(-8,-3,100)
print("Radius of the plain: ", round(radius,2))
print("Rydberg Blockade:", round(r_b*2,2))
print("Dichte: ", np.round((number_atoms*np.pi*r_b*r_b)/(np.pi*radius**2),2))


# In[22]:


#%% Generating folders and counting configurations

folder_a= "/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Configurations/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)
if number_atoms>1000 and density>=0.5:
    if not os.path.exists(folder_a):
        os.makedirs(folder_a)
        print(folder_a +" erstellt")
    else:
        path, dirs, files = next(os.walk(folder_a))
        speicher_atome = len(files)
else:
    speicher_atome=0

    
print("# files in Ordner atom_"+str(number_atoms)+": ", speicher_atome)
print("speicher_atome:  ",speicher_atome)


# In[23]:


number_calculations=1


# In[24]:


global save_nr
save_nr = 0


start=time.time()
print("iteration: 0")

atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr)
eigenvalues, eigenvectors = generate_hamiltonian(atoms_array)
probability=Berechnung(eigenvalues,eigenvectors,number_atoms,time_array)


r_1,r_2,r_1_inf,r_2_inf,deviation,deviation_inf,density,density_inf,ipr,ipr_inf,n_intervall_inf= auswertung(probability)

density_inf_S               = 0     
density_S                   = 0  
ipr_S                       = 0
ipr_inf_S                   = 0          
deviation_S                 = 0                    
deviation_inf_S             = 0         
r_1_inf_S                   = 0   
r_2_inf_S                   = 0  
r_1_S                       = 0
r_2_S                       = 0



save_nr+=1
for i in range(0,number_calculations-1):
    print("Iteration:", i+1)
    
    atoms_array= find_atoms_file(number_atoms, density, speicher_atome, folder_a, save_nr)
    eigenvalues, eigenvectors = generate_hamiltonian(atoms_array)
    probability=Berechnung(eigenvalues,eigenvectors,number_atoms,time_array)


    r_1_tmp,r_2_tmp,r_1_inf_tmp,r_2_inf_tmp,deviation_tmp,deviation_inf_tmp,density_tmp,density_inf_tmp,ipr_tmp,ipr_inf_tmp,n_intervall_inf_tmp= auswertung(probability)

   
    density_inf_prev              = density_inf
    density_prev                  = density
    ipr_prev                      = ipr
    ipr_inf_prev                  = ipr_inf
    deviation_prev                = deviation
    deviation_inf_prev            = deviation_inf
    r_1_inf_prev                  = r_1_inf
    r_2_inf_prev                  = r_2_inf
    r_1_prev                      = r_1
    r_2_prev                      = r_2
    
    density_inf                      = (density_inf*(i+1)+density_inf_tmp)/(i+2)
    density                          = (density*(i+1)+density_tmp)/(i+2)
    ipr                              = (ipr*(i+1)+ipr_tmp)/(i+2)
    ipr_inf                          = (ipr_inf*(i+1)+ipr_inf_tmp)/(i+2)
    deviation                        = (deviation*(i+1)+deviation_tmp)/(i+2)
    deviation_inf                    = (deviation_inf*(i+1)+deviation_inf_tmp)/(i+2)
    r_1_inf                          = (r_1_inf*(i+1)+r_1_inf_tmp)/(i+2)
    r_2_inf                          = (r_2_inf*(i+1)+r_2_inf_tmp)/(i+2)
    r_1                              = (r_1*(i+1)+r_1_tmp)/(i+2)
    r_2                              = (r_2*(i+1)+r_2_tmp)/(i+2)
    
    density_inf_S              += (density_inf_tmp-density_inf)*(density_inf_tmp-density_inf_prev)
    density_S                  += (density_tmp-density)*(density_tmp-density_prev)
    ipr_S                      += (ipr_tmp-ipr)*(ipr_tmp-ipr_prev)
    ipr_inf_S                  += (ipr_inf_tmp-ipr_inf)*(ipr_inf_tmp-ipr_inf_prev)
    deviation_S                += (deviation_tmp-deviation)*(deviation_tmp-deviation_prev)
    deviation_inf_S            += (deviation_inf_tmp-deviation_inf)*(deviation_inf_tmp-deviation_inf_prev)
    r_1_inf_S                  += (r_1_inf_tmp-r_1_inf)*(r_1_inf_tmp-r_1_inf_prev)
    r_2_inf_S                  += (r_2_inf_tmp-r_2_inf)*(r_2_inf_tmp-r_2_inf_prev)
    r_1_S                      += (r_1_tmp-r_1)*(r_1_tmp-r_1_prev)
    r_2_S                      += (r_2_tmp- r_2)*( r_2_tmp- r_2_prev)
    

    save_nr +=1

    
density_inf_S             =  np.sqrt(density_inf_S/number_calculations)
density_S                 =  np.sqrt(density_S/number_calculations)
ipr_S                     =  np.sqrt(ipr_S/number_calculations)
ipr_inf_S                 =  np.sqrt(ipr_inf_S /number_calculations)
deviation_S               =  np.sqrt(deviation_S/number_calculations)
deviation_inf_S           =  np.sqrt(deviation_inf_S/number_calculations)
r_1_inf_S                 =  np.sqrt(r_1_inf_S/number_calculations)
r_2_inf_S                 =  np.sqrt(r_2_inf_S/number_calculations)
r_1_S                     =  np.sqrt(r_1_S/number_calculations)
r_2_S                     =  np.sqrt(r_2_S/number_calculations)    


data = open("/pfs/data2/home/hd/hd_hd/hd_wo455/Schreibtisch/Results/Transport/density_"+str(np.round(density,3))+"/atoms_"+str(number_atoms)+".npy","wb")
np.save(data,density_inf)
np.save(data,density)
np.save(data,ipr)
np.save(data,ipr_inf)
np.save(data,deviation)
np.save(data,deviation_inf)
np.save(data,r_1_inf)
np.save(data,r_2_inf)
np.save(data,r_1)
np.save(data,r_2)


np.save(data,density_inf_S)
np.save(data,density_S)
np.save(data,ipr_S)
np.save(data,ipr_inf_S)
np.save(data,deviation_S)
np.save(data,deviation_inf_S)
np.save(data,r_1_inf_S)
np.save(data,r_2_inf_S)
np.save(data,r_1_S)
np.save(data,r_2_S)

data.close()

print(time.time()-start)






