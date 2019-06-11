# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:07:46 2019

# Takes PDB file and outputs full maps for different atomtypes
# Code 1

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
import os

# Garde Group Scripts:
import voxelize_functions
from voxelize_functions import *
import atom_types
from atom_types import atom_id


from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)


start = time.time()

# Settings:-------------------------------------------------------------------

# Defaults:
cutoff = 10 # Cutoff (beyond which a point cannot feel an atom) (angstroms)
std = 1.0 #standard deviation for gaussian
dx = 0.5 # voxel size
buffer = 20 # Buffer size around edges of protein in angstroms
n_atomtypes = 4 # Number of types of atoms, must be aligned with atom_types.py

# Initialize:
proc_file = 0 #number of files processed

# Identify file locations:
curr_path = os.getcwd()
# PDB files are in a subdirectory named "2clean_pdb"
# Pickle files are in a subdirectory named "pickle"
pdb_path = os.path.join(curr_path,'2clean_pdb')
pickle_path = os.path.join(curr_path,'3pickle_perpdb')
all_files = os.listdir(pdb_path)

# Load in PDB files:-----------------------------------------------------------

for item in all_files:
    file, extension = os.path.splitext(item)
    if ((extension == '.pdb')&(file[-6:]=='-clean')):
        proc_file +=1
        
        print('Processing File', proc_file, file)
        
        structure_id = file
        filename = os.path.join(pdb_path,item)
        structure = parser.get_structure(structure_id,filename)
       
## Populate a grid with atomic densities:--------------------------------------
        # Define grid edges
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

        gridx, gridy, gridz = np.meshgrid(linx,liny,linz)
        gridshape = np.shape(gridx)
        
        # Fill densities into grid
        
        occupancy = np.zeros([n_atomtypes,np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
        
        for residue in list(structure.get_residues())[0:10]: # Limited to residues 1:10 for time only
            print('Assessing Residue', residue)
            for atom in residue.get_list():
                id_mat = atom_id(atom)
                for i in range(0,n_atomtypes):
                    if id_mat[i]==1:
                        atomcoord = atom.get_coord()
                        for x in np.where(abs(linx-atomcoord[0])<cutoff/2.0)[0]:
                            for y in np.where(abs(liny-atomcoord[1])<cutoff/2.0)[0]:
                                for z in np.where(abs(linz-atomcoord[2])<cutoff/2.0)[0]:
                                     pointcoord = np.array([linx[x],liny[y],linz[z]])
                                     occupancy[i,x,y,z] +=atom_density(np.linalg.norm(pointcoord-atomcoord),std)
        pname = structure_id + '.pickle'
        picklename = os.path.join(pickle_path,pname)    
        pickle_out = open(picklename,"wb")
        pickle.dump(occupancy, pickle_out)
        pickle_out.close()

print('Unprocessed files: ',(len(all_files)-proc_file))

end = time.time()

print('Time Elapsed:',end-start)
