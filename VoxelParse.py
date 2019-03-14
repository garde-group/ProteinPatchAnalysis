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


from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)


start = time.time()

# Load in PDB file:-----------------------------------------------------------
structure_id = "fab"
filename = "fabA.pdb"
structure = parser.get_structure(structure_id,filename)


# Settings:-------------------------------------------------------------------
cutoff = 10 # Cutoff (beyond which a point cannot feel an atom) (angstroms)
std = 1.0 #standard deviation for gaussian
dx = 0.5 # voxel size
side_length = 10 # patch size in angstroms


# Populate a grid with atomic densities:--------------------------------------
grid_min = np.array([-21.5,-58.9,-18.6])-20
grid_max = np.array([11.8,3.6,44.5])+20

linx = np.arange(grid_min[0],grid_max[0],dx)
liny = np.arange(grid_min[1],grid_max[1],dx)
linz = np.arange(grid_min[2],grid_max[2],dx)

gridx, gridy, gridz = np.meshgrid(linx,liny,linz)
gridshape = np.shape(gridx)
carbon_density = np.zeros([np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
nitrogen_density = np.zeros([np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
oxygen_density = np.zeros([np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])
sulfur_density = np.zeros([np.shape(linx)[0],np.shape(liny)[0],np.shape(linz)[0]])

for residue in structure[0][' '].get_list():
    print('Assessing Residue', residue)
    for atom in residue.get_list():
        #print(atom.get_name()[0])
        if atom.get_name()[0]=='C':
            atomcoord = atom.get_coord()
            for x in np.where(abs(linx-atomcoord[0])<side_length/2.0)[0]:
                for y in np.where(abs(liny-atomcoord[1])<side_length/2.0)[0]:
                    for z in np.where(abs(linz-atomcoord[2])<side_length/2.0)[0]:
                        pointcoord = np.array([linx[x],liny[y],linz[z]])
                        carbon_density[x,y,z] += atom_density(np.linalg.norm(pointcoord-atomcoord),std)
                        
        elif atom.get_name()[0]=='N':
            atomcoord = atom.get_coord()
            for x in np.where(abs(linx-atomcoord[0])<side_length/2.0)[0]:
                for y in np.where(abs(liny-atomcoord[1])<side_length/2.0)[0]:
                    for z in np.where(abs(linz-atomcoord[2])<side_length/2.0)[0]:
                        pointcoord = np.array([linx[x],liny[y],linz[z]])
                        nitrogen_density[x,y,z] += atom_density(np.linalg.norm(pointcoord-atomcoord),std)

        elif atom.get_name()[0]=='O':
            atomcoord = atom.get_coord()
            for x in np.where(abs(linx-atomcoord[0])<side_length/2.0)[0]:
                for y in np.where(abs(liny-atomcoord[1])<side_length/2.0)[0]:
                    for z in np.where(abs(linz-atomcoord[2])<side_length/2.0)[0]:
                        pointcoord = np.array([linx[x],liny[y],linz[z]])
                        oxygen_density[x,y,z] += atom_density(np.linalg.norm(pointcoord-atomcoord),std)
                        
        elif atom.get_name()[0]=='S':
            atomcoord = atom.get_coord()
            for x in np.where(abs(linx-atomcoord[0])<side_length/2.0)[0]:
                for y in np.where(abs(liny-atomcoord[1])<side_length/2.0)[0]:
                    for z in np.where(abs(linz-atomcoord[2])<side_length/2.0)[0]:
                        pointcoord = np.array([linx[x],liny[y],linz[z]])
                        sulfur_density[x,y,z] += atom_density(np.linalg.norm(pointcoord-atomcoord),std)
             
        elif atom.get_name()[0]=='H':
            atomcoord = atom.get_coord()
            
        else:
            print('WARNINGWARNINGWARNING: Unknown Atom:',atom.get_name()[0])
        

                
    
pickle_out = open("voxel.pickle","wb")
pickle.dump([carbon_density,nitrogen_density,oxygen_density,sulfur_density], pickle_out)
pickle_out.close()


end = time.time()

print('Time Elapsed:',end-start)

