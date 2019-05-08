# coding: utf-8

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

from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)

# Calculate center of geometry for a given residue
def res_cog(residue):
    coord = [residue.get_list()[i].get_coord() for i in range(0,np.shape(residue.get_list())[0])]
    cog = np.mean(coord,axis=0)
    return cog

start = time.time()

# Settings
cutoff = 10 # Cutoff (beyond which a point cannot feel an atom) (angstroms)
std = 1.0 #standard deviation for gaussian
dx = 0.5 # voxel size
side_length = 10 # patch size in angstroms

curr_path = os.getcwd()
# assume that pdb files are in a subdirectory named "pdb"
pdb_path = os.path.join(curr_path,'pdb')
pickle_path = os.path.join(curr_path,'pickle')
print(pdb_path)
print(pickle_path)
all_files = os.listdir(pdb_path)
for item in all_files:
    file, extension = os.path.splitext(item)
    if (extension == '.pdb'):

# Part 1        
        
        structure_id = file
        filename = os.path.join(pdb_path,item)
        structure = parser.get_structure(structure_id,filename)
        print(structure_id)
        print(filename)
        # Populate a grid with atomic densities
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

# Part 2        
                            
        side_length = 25 #0,.5 Number of voxels in patch
        
        # Combine all densities into one matrix
        all_filters = np.stack([carbon_density,oxygen_density,nitrogen_density,sulfur_density])
        n_filters = 4
        PatchesAllColors = np.zeros([1,229,49,49,49])
        
        ## Define patches
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

        pname = structure_id + '.pickle'
        picklename = os.path.join(pickle_path,pname)    
        pickle_out = open(picklename,"wb")
        pickle.dump(PatchesAllColors, pickle_out)
        pickle_out.close()

end = time.time()

print('Time Elapsed:',end-start)