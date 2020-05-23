
import numpy as np
import math
import random
import scipy
import time

#%% 2D radial symmetric distributed
def to_ind(coord, radius):
    x_shifted = coord[0]+radius
    y_shifted = np.abs(coord[1]-radius)

    j = np.floor(x_shifted/(size_square))
    i = np.floor(y_shifted/(size_square))
    return int(i),int(j)

def to_coord(ind,radius):
    x = ind[1]*size_square
    y = ind[0]*size_square
    
    x_shifted = x -radius
    y_shifted = radius - y
 
    return [x_shifted, y_shifted]


def get_atoms_near(punkt, counting_atoms,radius):
    m,n = to_ind(punkt[0],radius)
    atoms_near=[]
    atoms_near.extend(counting_atoms[m][n])

    try:
        atoms_near.extend(counting_atoms[m+1][n])             
        atoms_near.extend(counting_atoms[m+1][n+1])       
        atoms_near.extend(counting_atoms[m][n+1])
        atoms_near.extend(counting_atoms[m-1][n-1])        
        atoms_near.extend(counting_atoms[m-1][n])
        atoms_near.extend(counting_atoms[m-1][n+1])
        atoms_near.extend(counting_atoms[m][n-1])
        atoms_near.extend(counting_atoms[m+1][n-1])
        return atoms_near
    except IndexError:
        if m+1<number_squares:
            atoms_near.extend(counting_atoms[m+1][n])
            if n+1< number_squares:
                atoms_near.extend(counting_atoms[m+1][n+1])
        if n+1<number_squares:
            atoms_near.extend(counting_atoms[m][n+1])

        if (m-1)>=0 and (n-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n-1])
        if (m-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n])
            if n+1<number_squares:
                atoms_near.extend(counting_atoms[m-1][n+1])
        if (n-1)>=0: 
            atoms_near.extend(counting_atoms[m][n-1])
            if m+1<number_squares:
                atoms_near.extend(counting_atoms[m+1][n-1])
    return atoms_near


def produce_atoms(number_atoms, radius, r_b, versuch_max):
    global size_square
    global number_squares
    size_square = 2*r_b   #np.sqrt(density*50*r_b**2)   #2*r_b   #wähle size_square so, dass 50 atome drin sind.
    #size_square gerade klein genug um 8 Felder außen rum zu benutzen um abzugleichen
    number_squares= int(radius*2/size_square)
    size_square = radius*2/number_squares
  
    
    counting_atoms = [[[] for i in range(number_squares)] for i in range(number_squares)]
        
    atoms = [[0,0]]
    m,n = to_ind([0,0], radius)
    counting_atoms[m][n] = [[0,0]]
    
    i=1
    while i <number_atoms:
        r=(radius-r_b)*np.sqrt(random.uniform(0,1))
        theta=random.uniform(0,1)*2*np.pi
        x,y=r * np.cos(theta), r * np.sin(theta)
        punkt = np.array([[x,y]]) 
        gesetzt= False
        
        k=0
        while (gesetzt==False):
            k+=1
            if k>versuch_max:
                atoms=[[0,0]]
                counting_atoms=[[[] for i in range(number_squares)] for i in range(number_squares)]
                m,n = to_ind([0,0], radius)
                counting_atoms[m][n] = [[0,0]]
                i=1
                                  
            gesetzt=True
            atoms_near = get_atoms_near(punkt, counting_atoms,radius)

            if atoms_near:
                if ((np.min(scipy.spatial.distance.cdist(atoms_near, punkt, metric='euclidean')))<(2*r_b)):
                    r=(radius-r_b)*np.sqrt(random.uniform(0,1))
                    theta=random.uniform(0,1)*2*np.pi
                    x,y=r * np.cos(theta), r * np.sin(theta)
                    punkt = np.array([[x,y]]) 
                    gesetzt=False
    
        atoms.append(punkt[0])
        m,n = to_ind(punkt[0],radius)
        counting_atoms[m][n].append(punkt[0].tolist())
        i+=1

    return np.round(atoms,4)



def produce_2D_gaus(number_atoms, radius, r_b,variance):

    global size_square
    global number_squares
    size_square = 10*r_b #np.sqrt(density*50*r_b**2) #für 50 atome   #2*r_b   
    number_squares= int(radius*2/size_square)
    size_square = radius*2/number_squares
    
    counting_atoms = [[[0,[]] for i in range(number_squares)] for i in range(number_squares)]
        
    atoms = [[0,0]]
    m,n = to_ind([0,0],radius)
    counting_atoms[m][n][0]+=1
    counting_atoms[m][n][1] = [[0,0]]

    
    mean = [0, 0]
    cov = variance
    
    atoms_generated=10*number_atoms
    gaus_atoms=np.zeros((atoms_generated,2))
    gaus_atoms[:,0], gaus_atoms[:,1] = np.random.multivariate_normal(mean, cov, atoms_generated).T
    gaus_atoms=gaus_atoms[(scipy.spatial.distance.cdist(gaus_atoms, [[0,0]], metric='euclidean')<radius)[:,0]]
    anzahl_gaus=len(gaus_atoms)
    
    j=0
    
    for i in range(number_atoms-len(atoms)):
        
        punkt = np.array([gaus_atoms[j]]) 
        gesetzt= False
    
        while (gesetzt==False):
            j+=1
            gesetzt=True
            
            if j>=anzahl_gaus-1:
                gaus_atoms=np.zeros((atoms_generated,2))
                gaus_atoms[:,0], gaus_atoms[:,1] = np.random.multivariate_normal(mean, cov, atoms_generated).T
                gaus_atoms=gaus_atoms[(scipy.spatial.distance.cdist(gaus_atoms, [[0,0]], metric='euclidean')<radius)[:,0]]
                anzahl_gaus=len(gaus_atoms)
                print("jo")
                j=0
                
            atoms_near = get_atoms_near(punkt, counting_atoms,radius)

            if atoms_near:
                if ((np.min(scipy.spatial.distance.cdist(atoms_near, punkt, metric='euclidean')))<(2*r_b)):
                    punkt = np.array([gaus_atoms[j]]) 
                    gesetzt=False   
                    
        j+=1
                    
        atoms.append(punkt[0])
        m,n = to_ind(punkt[0],radius)
        counting_atoms[m][n][0]+=1
        counting_atoms[m][n][1].append(punkt[0].tolist())

    return np.round(atoms,4)





def hexagonal(r_tmp,r_b):
    atoms =[]
    spalten=math.ceil(2*r_tmp/r_b)

    x=-r_tmp
    y= r_tmp
    s=-1
    for i in range(spalten):
        for j in range(spalten):
            if np.sqrt(x*x+y*y)<(r_tmp):
                atoms.append([x,y])
            y-=2*r_b
        x+=np.sqrt((2*r_b)**2-r_b**2)
        if s >0:
            y=r_tmp
        else:
            y=r_tmp-r_b
        s*=-1
    return np.array(atoms)


def square(r_tmp,r_b):
    atoms =[]
    spalten=math.ceil(2*r_tmp/r_b)

    x=-r_tmp
    y= r_tmp

    for i in range(spalten):
        for j in range(spalten):
            if np.sqrt(x*x+y*y)<(r_tmp):
                atoms.append([x,y])
            y-=2*r_b
        x+=2*r_b
   
        y=r_tmp
     
    return np.array(atoms)


#%% 3D 
    
def to_ind_3D(coord,radius):
    x_shifted = coord[0]+radius
    y_shifted = np.abs(coord[1]-radius)
    z_shifted = coord[2]+radius

    j = np.floor(x_shifted/(size_square))
    i = np.floor(y_shifted/(size_square))
    k = np.floor(z_shifted/(size_square))
    
    return int(i),int(j), int(k)

def to_coord_3D(ind,radius):
    x = ind[1]*size_square
    y = ind[0]*size_square
    z = ind[2]*size_square
    
    x_shifted = x -radius
    y_shifted = radius - y
    z_shifted = z - radius
 
    return [x_shifted, y_shifted, z_shifted]


def get_atoms_near_3D(punkt, counting_atoms,radius):
    m,n,l = to_ind_3D(punkt[0],radius)
    atoms_near=[]
    atoms_near.extend(counting_atoms[m][n][l][1])

    try:
        atoms_near.extend(counting_atoms[m+1][n][l][1])             
        atoms_near.extend(counting_atoms[m+1][n+1][l][1])       
        atoms_near.extend(counting_atoms[m][n+1][l][1])
        atoms_near.extend(counting_atoms[m-1][n-1][l][1])        
        atoms_near.extend(counting_atoms[m-1][n][l][1])
        atoms_near.extend(counting_atoms[m-1][n+1][l][1])
        atoms_near.extend(counting_atoms[m][n-1][l][1])
        atoms_near.extend(counting_atoms[m+1][n-1][l][1])
        
        #z größer
        atoms_near.extend(counting_atoms[m][n][l+1][1])  
        atoms_near.extend(counting_atoms[m+1][n][l+1][1])             
        atoms_near.extend(counting_atoms[m+1][n+1][l+1][1])       
        atoms_near.extend(counting_atoms[m][n+1][l+1][1])
        atoms_near.extend(counting_atoms[m-1][n-1][l+1][1])        
        atoms_near.extend(counting_atoms[m-1][n][l+1][1])
        atoms_near.extend(counting_atoms[m-1][n+1][l+1][1])
        atoms_near.extend(counting_atoms[m][n-1][l+1][1])
        atoms_near.extend(counting_atoms[m+1][n-1][l+1][1])
        
        #z kleiner
        atoms_near.extend(counting_atoms[m][n][l-1][1])  
        atoms_near.extend(counting_atoms[m+1][n][l-1][1])             
        atoms_near.extend(counting_atoms[m+1][n+1][l-1][1])       
        atoms_near.extend(counting_atoms[m][n+1][l-1][1])
        atoms_near.extend(counting_atoms[m-1][n-1][l-1][1])        
        atoms_near.extend(counting_atoms[m-1][n][l-1][1])
        atoms_near.extend(counting_atoms[m-1][n+1][l-1][1])
        atoms_near.extend(counting_atoms[m][n-1][l-1][1])
        atoms_near.extend(counting_atoms[m+1][n-1][l-1][1])
        
        return atoms_near
    except IndexError:
        
        #für z normal
        if m+1<number_squares:
            atoms_near.extend(counting_atoms[m+1][n][l][1])
            if n+1< number_squares:
                atoms_near.extend(counting_atoms[m+1][n+1][l][1])
        if n+1<number_squares:
            atoms_near.extend(counting_atoms[m][n+1][l][1])

        if (m-1)>=0 and (n-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n-1][l][1])
        if (m-1)>=0: 
            atoms_near.extend(counting_atoms[m-1][n][l][1])
            if n+1<number_squares:
                atoms_near.extend(counting_atoms[m-1][n+1][l][1])
        if (n-1)>=0: 
            atoms_near.extend(counting_atoms[m][n-1][l][1])
            if m+1<number_squares:
                atoms_near.extend(counting_atoms[m+1][n-1][l][1])
               
        #für z eins größer
        if l+1<number_squares:
            atoms_near.extend(counting_atoms[m][n][l+1][1]) 
            if m+1<number_squares:
                atoms_near.extend(counting_atoms[m+1][n][l+1][1])
                if n+1< number_squares:
                    atoms_near.extend(counting_atoms[m+1][n+1][l+1][1])
            if n+1<number_squares:
                atoms_near.extend(counting_atoms[m][n+1][l+1][1])

            if (m-1)>=0 and (n-1)>=0: 
                atoms_near.extend(counting_atoms[m-1][n-1][l+1][1])
            if (m-1)>=0: 
                atoms_near.extend(counting_atoms[m-1][n][l+1][1])
                if n+1<number_squares:
                    atoms_near.extend(counting_atoms[m-1][n+1][l+1][1])
            if (n-1)>=0: 
                atoms_near.extend(counting_atoms[m][n-1][l+1][1])
                if m+1<number_squares:
                    atoms_near.extend(counting_atoms[m+1][n-1][l+1][1])
            
        #für z eins kleiner
        if (l-1)>=0:
            atoms_near.extend(counting_atoms[m][n][l-1][1]) 
            if m+1<number_squares:
                atoms_near.extend(counting_atoms[m+1][n][l-1][1])
                if n+1< number_squares:
                    atoms_near.extend(counting_atoms[m+1][n+1][l-1][1])
            if n+1<number_squares:
                atoms_near.extend(counting_atoms[m][n+1][l-1][1])

            if (m-1)>=0 and (n-1)>=0: 
                atoms_near.extend(counting_atoms[m-1][n-1][l-1][1])
            if (m-1)>=0: 
                atoms_near.extend(counting_atoms[m-1][n][l-1][1])
                if n+1<number_squares:
                    atoms_near.extend(counting_atoms[m-1][n+1][l-1][1])
            if (n-1)>=0: 
                atoms_near.extend(counting_atoms[m][n-1][l-1][1])
                if m+1<number_squares:
                    atoms_near.extend(counting_atoms[m+1][n-1][l-1][1])
    
    return atoms_near


def produce_3D_gaus(number_atoms, radius, r_b,variance):

    global size_square
    global number_squares
    size_square = 10*r_b #np.sqrt(density*50*r_b**2) #für 50 atome   #2*r_b   
    number_squares= int(radius*2/size_square)
    size_square = radius*2/number_squares
    
    counting_atoms = [[[[0,[]] for i in range(number_squares)] for i in range(number_squares)] for i in range(number_squares)]
        
    atoms = [[0,0,0]]
    m,n,l = to_ind_3D([0,0,0],radius)
    counting_atoms[m][n][l][0]+=1
    counting_atoms[m][n][l][1] = [[0,0,0]]

    
    mean = [0, 0,0]
    cov = variance
    
    atoms_generated=10*number_atoms
    gaus_atoms=np.zeros((atoms_generated,3))
    gaus_atoms[:,0], gaus_atoms[:,1], gaus_atoms[:,2] = np.random.multivariate_normal(mean, cov, atoms_generated).T
    gaus_atoms=gaus_atoms[(scipy.spatial.distance.cdist(gaus_atoms, [[0,0,0]], metric='euclidean')<radius)[:,0]]
    anzahl_gaus=len(gaus_atoms)
    
    j=0
    
    for i in range(number_atoms-len(atoms)):
        
        punkt = np.array([gaus_atoms[j]]) 
        gesetzt= False
    
        while (gesetzt==False):
            j+=1
            gesetzt=True
            
            if j>=anzahl_gaus-1:
                gaus_atoms=np.zeros((atoms_generated,3))
                gaus_atoms[:,0], gaus_atoms[:,1], gaus_atoms[:,2] = np.random.multivariate_normal(mean, cov, atoms_generated).T
                gaus_atoms=gaus_atoms[(scipy.spatial.distance.cdist(gaus_atoms, [[0,0,0]], metric='euclidean')<radius)[:,0]]
                anzahl_gaus=len(gaus_atoms)
                print("jo")
                
                j=0
                
            atoms_near = get_atoms_near_3D(punkt, counting_atoms,radius)

            if atoms_near:
                if ((np.min(scipy.spatial.distance.cdist(atoms_near, punkt, metric='euclidean')))<(2*r_b)):
                    punkt = np.array([gaus_atoms[j]]) 
                    gesetzt=False   
                    
        j+=1
                    
        atoms.append(punkt[0])
        m,n,l = to_ind_3D(punkt[0],radius)
        counting_atoms[m][n][l][0]+=1
        counting_atoms[m][n][l][1].append(punkt[0].tolist())

    return np.round(atoms,4)




def produce_3D_radialsym(number_atoms, radius, r_b):
    global size_square
    global number_squares
    size_square = 10*r_b #np.sqrt(density*50*r_b**2)   #2*r_b   #wähle size_square so, dass 50 atome drin sind.
    #size_square gerade klein genug um 8 Felder außen rum zu benutzen um abzugleichen
    number_squares= int(radius*2/size_square)
    size_square = radius*2/number_squares
    
    
    
    counting_atoms = [[[[0,[]] for i in range(number_squares)] for i in range(number_squares)] for i in range(number_squares)]
        
    atoms = [[0,0,0]]
    m,n,l = to_ind_3D([0,0,0], radius)
    counting_atoms[m][n][l][0]+=1
    counting_atoms[m][n][l][1] = [[0,0,0]]
    
        
    for i in range(number_atoms-len(atoms)):
        
        r=(radius-r_b)*math.pow(random.uniform(0,1),1/3)
        phi=random.uniform(0,1)*2*np.pi
        theta = random.uniform(0,1)*np.pi
        
        x = r * np.sin( theta) * np.cos( phi )
        y = r * np.sin( theta) * np.sin( phi )
        z = r * np.cos( theta )

        punkt = np.array([[x,y,z]]) 
        gesetzt= False
        
        while (gesetzt==False):
            gesetzt=True

            atoms_near = get_atoms_near_3D(punkt, counting_atoms,radius)

            if atoms_near:
                if ((np.min(scipy.spatial.distance.cdist(atoms_near, punkt, metric='euclidean')))<(2*r_b)):
                    r=(radius-r_b)*math.pow(random.uniform(0,1),1/3)
                    phi=random.uniform(0,1)*2*np.pi
                    theta = random.uniform(0,1)*np.pi
                    
                    x = r * np.sin( theta) * np.cos( phi )
                    y = r * np.sin( theta) * np.sin( phi )
                    z = r * np.cos( theta )
            
                    punkt = np.array([[x,y,z]]) 
                    gesetzt=False
                              
        atoms.append(punkt[0])
        m,n,l = to_ind_3D(punkt[0],radius)
        counting_atoms[m][n][l][0]+=1
        counting_atoms[m][n][l][1].append(punkt[0].tolist())

    return np.round(atoms,4)


