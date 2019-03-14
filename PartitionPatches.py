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
from mpl_toolkits.mplot3d import Axes3D

side_length = 25 #0,.5 Number of voxels in patch



# Combine all densities into one matrix
all_filters = np.stack([carbon_density,oxygen_density,nitrogen_density,sulfur_density])
n_filters = 4
PatchesAllColors = np.zeros([1,229,49,49,49])

## Define patches:-------------------------------------------------------------
## The center of geometry of each residue constitutes the center of a patch:

for filter in all_filters:
    # i is counter for determining progress
    i=0
    
    Patches= np.zeros([49,49,49]) # Number of voxels in patch
    
    # Identify first residue:
    residue = structure[0][' '].get_list()[0]
    
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
    
    patch=filter[x_start:x_end,y_start:y_end,z_start:z_end]
    Patches=np.stack([Patches, patch])
    
    # Loop over all other residues:
    for residue in structure[0][' '].get_list():
        i+=1
        print('Analyzing residue',i)
        
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
        
        
        patch=filter[x_start:x_end,y_start:y_end,z_start:z_end]
        patch=np.reshape(patch,(1,49,49,49))
        Patches=np.concatenate([Patches, patch],axis=0)
    
    Patches = np.reshape(Patches,(1,229,49,49,49))
    PatchesAllColors = np.concatenate([PatchesAllColors,Patches],axis=0)
    
PatchesAllColors = np.swapaxes(PatchesAllColors,0,1)
# Axis convention: 
# 1: Sample axis
# 2: Color axis (atom identity)
# 3-5: Cartesian axes


#        

                