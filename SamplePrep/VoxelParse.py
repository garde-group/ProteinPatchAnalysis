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
from atom_types_pdbqt import atom_id_pdbqt


from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)


start = time.time()

# Settings:-------------------------------------------------------------------

# Defaults:
cutoff = 10 # Cutoff (beyond which a point cannot feel an atom) (angstroms)
std = 1.0 #standard deviation for gaussian
dx = 0.5 # voxel size
buffer = 20 # Buffer size around edges of protein in angstroms
n_atomtypes = 10 # Number of types of atoms, must be aligned with atom_types.py

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
        # Note that this section requires installation of MGLtools and pythonsh needs to be stored at the following location:
        pdbqtfile = pdb_path +'/'+file+'.pdbqt'
        os.system('/disk1/ligdesign/MGLTools/mgltools_i86Linux2_1.5.2/bin/pythonsh prepare_receptor4.py -r '+pdb_path+'/'+file+'.pdb -o ' + pdb_path +'/'+file+'.pdbqt')
        os.system('grep ATOM '+pdbqtfile+'>'+pdb_path+'/temp')
        os.system('sed \'s/^.\{60\}/& /\' '+pdb_path+ '/temp > '+ pdbqtfile)
        #os.system('mv '+pdb_path+ '/temp '+pdbqtfile)
        pdbqt = read_data(pdbqtfile)
        
        print('Processing File', proc_file, file)
        
        structure_id = file
        filename = os.path.join(pdb_path,item)
        structure = parser.get_structure(structure_id,pdbqtfile) #filename)
       
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
        
        for residue in list(structure.get_residues()): # Limited to residues 1:10 for time only
            print('Assessing Residue', residue)
            for atom in residue.get_list():
                #print(atom.serial_number)
                atom_name = pdbqt.iloc[np.where(pdbqt[1].astype(float)==atom.serial_number)[0]].values[0][-1]
                id_mat = atom_id_pdbqt(atom_name)
                #print(pdbqt.loc[atom.serial_number-1][12])
                #print(id_mat)
                if np.sum(id_mat)!=1.0:
                    print('ERROR: Encountered Undocumented Atom Type: ',atom_name)
                    print(pdbqt.loc[atom.serial_number-1])

            
                #id_mat = atom_id_pdbqt(atom)
                #print(id_mat)
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
