# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 20:07:36 2019

Takes atom maps from VoxelParse.py and partitions them into patches
Code 2

@author: camil
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as col
from scipy.optimize import curve_fit
import pandas as pd
from math import factorial
import random
import time
import Bio
import pickle
import os
from mpl_toolkits.mplot3d import Axes3D

import voxelize_functions
from voxelize_functions import *


start = time.time()

# Settings:-------------------------------------------------------------------
side_length = 25 # Number of voxels in patch
full_length = side_length*2-1
n_atomtypes = 4 # Number of types of atoms, must be aligned with atom_types.py
buffer = 20 # Buffer size around edges of protein in angstroms
dx = 0.5 # voxel size
exposure_cutoff = 0.3 # Fractional SASA exposure above which patches are included

Patches = np.zeros([1,n_atomtypes,full_length,full_length,full_length])

# Oneliner to read xvg file ====================================
def read_data(fname):
    data = [ [i for i in line.split()]  
             for line in open(fname).readlines() 
             if line[0] not in ['#','@'] and len(line.rstrip()) > 0 and 
             line.split()[0][0] not in ['#', '@'] ]
    data = pd.DataFrame(data)
    return data


# Identify file locations:
curr_path = os.getcwd()
# PDB files are in a subdirectory named "2clean_pdb"
# Pickle files are in a subdirectory named "3pickle_perpdb"
pdb_path = os.path.join(curr_path,'2clean_pdb')
pickle_path = os.path.join(curr_path,'3pickle_perpdb')
combined_path = os.path.join(curr_path,'4pickle_combined')
all_pickles = os.listdir(pickle_path)


for item in all_pickles: 
    file, extension = os.path.splitext(item)
    if (extension == '.pickle'):
        
        # For every structure load in occupancy grid and structure:
        pickle_in = open(pickle_path+'/'+item,"rb")
        occupancy = pickle.load(pickle_in)
        structure = parser.get_structure(file,pdb_path+'/'+file+'.pdb')
        
        
        # Define grid edges from structure:
        coord_list = [atom.coord for atom in structure.get_atoms()]
        
        xmin = min([coord_list[i][0] for i in range(0,np.shape(coord_list)[0])])
        xmin = xmin-buffer
        xmax = max([coord_list[i][0] for i in range(0,np.shape(coord_list)[0])])
        xmax = xmax+buffer
        
        ymin = min([coord_list[i][1] for i in range(0,np.shape(coord_list)[0])])
        ymin = ymin-buffer
        ymax = max([coord_list[i][1] for i in range(0,np.shape(coord_list)[0])])
        ymax = ymax+ buffer
        
        zmin = min([coord_list[i][2] for i in range(0,np.shape(coord_list)[0])])
        zmin = zmin-buffer
        zmax = max([coord_list[i][2] for i in range(0,np.shape(coord_list)[0])])
        zmax = zmax+buffer

        linx = np.arange(xmin,xmax,dx)
        liny = np.arange(ymin,ymax,dx)
        linz = np.arange(zmin,zmax,dx)        

        j=0
        for residue in list(structure.get_residues())[0:10]:
            j+=1
            
            exists = os.path.isfile(pickle_path+'/sasa-'+file+'.dat')
            if exists:
                res_area = read_data(pickle_path+'/sasa-'+file+'.dat')
                print('Analyzing residue',j)
                
                if float(res_area.values[j][3]) > exposure_cutoff:
                    # Calculate residue center:
                    center = res_cog(residue)
                    
                    # Calculate start/end positions:
                    vox_start = center-side_length/2.0
                    vox_end = center+side_length/2.0
                    
                    x_start=np.where((linx>vox_start[0])&(linx<vox_end[0]))[0][0]
                    x_end=np.where((linx>vox_start[0])&(linx<vox_end[0]))[0][-1]
                    
                    y_start=np.where((liny>vox_start[1])&(liny<vox_end[1]))[0][0]
                    y_end=np.where((liny>vox_start[1])&(liny<vox_end[1]))[0][-1]
                    
                    z_start=np.where((linz>vox_start[2])&(linz<vox_end[2]))[0][0]
                    z_end=np.where((linz>vox_start[2])&(linz<vox_end[2]))[0][-1]
                    
                    patch = occupancy[:,x_start:x_end,y_start:y_end,z_start:z_end]
                    patch = np.reshape(patch,(1,n_atomtypes,full_length,full_length,full_length))
                    Patches = np.concatenate([Patches,patch],axis=0)
            
pname = 'all_patches.pickle'
picklename = os.path.join(combined_path,pname)    
pickle_out = open(picklename,"wb")
pickle.dump(Patches, pickle_out)
pickle_out.close()

# Axis convention: 
# 1: Sample axis
# 2: Color axis (atom identity)
# 3-5: Cartesian axes      

    
end = time.time()

print('Time Elapsed:',end-start)
            