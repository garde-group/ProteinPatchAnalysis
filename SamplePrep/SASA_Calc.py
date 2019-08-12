#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 13:50:58 2019

@author: cbilodeau
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import matplotlib.colors as col
from scipy.optimize import curve_fit
import pandas as pd
from pandas import ExcelWriter
from math import factorial
import random
import time
import Bio
import pickle
from mpl_toolkits.mplot3d import Axes3D
import os

from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)

start = time.time()
proc_file=0



# Oneliner to read xvg file ====================================
def read_data(fname):
    data = [ [i for i in line.split()]  
             for line in open(fname).readlines() 
             if line[0] not in ['#','@'] and len(line.rstrip()) > 0 and 
             line.split()[0][0] not in ['#', '@'] ]
    data = pd.DataFrame(data)
    return data

# Read SASA Standard Data:
sasa_ref = read_data('sasa-ref.dat')
sasa_ref[1] = sasa_ref[1].astype(float)
sasa_dict = dict(sasa_ref.to_dict('split')['data'])

# Identify file locations:
curr_path = os.getcwd()
## PDB files and res_area.xvg are in a subdirectory named "2clean_pdb"
# Pickle files are in a subdirectory named "3pickle_perpdb"
pdb_path = os.path.join(curr_path,'2clean_pdb')
pickle_path = os.path.join(curr_path,'3pickle_perpdb')
all_files = os.listdir(pdb_path)




for item in all_files:
    file, extension = os.path.splitext(item)
    if ((extension == '.pdb')&(file[-6:]=='-clean')):
        exists = os.path.isfile(pdb_path+'/'+file+'-resarea.xvg')
        if exists:
            proc_file +=1
            
            print('Processing File', proc_file, file)
            # Read SASA data from pdb:
            #os.system('rm -rf temp')
            
            os.system('awk \'/^[^#|@]/{printf \"%f8\\n\", $2}\' '+pdb_path+'/'+file+'-resarea.xvg > temp1')
            #os.system('awk \'{if ($6!=a) print $4;a=$6}\' '+pdb_path+'/'+item+' >temp2')
            structure_clean = parser.get_structure(file,pdb_path+'/'+file+'.pdb')
            res_list = [residue.get_resname() for residue in list(structure_clean.get_residues())]
            #os.system('paste temp2 temp1 >temp3')
            res_area = read_data('temp1')
            #test2 = read_data('temp2')
            #res_area = read_data('temp3')
            res_area['resn_name'] = res_list 
            os.system('rm -rf temp*')
            
            res_area[0] = res_area[0].astype(float)
        
            # Calculate percent SASA:
            res_ref = [sasa_dict[res_area.values[i][1]] for i in range(0,np.shape(res_area.values)[0])]
        
     
            res_area['Ref'] = res_ref
            
            res_area['Percent'] = res_area[0]/res_area['Ref']
            decimals = pd.Series([2,2,2], index = [1, 'Ref', 'Percent'])
     
            res_area = res_area.round(decimals)
            
            res_area.to_csv(pickle_path+'/sasa-'+file+'.dat',header=None, index=None, sep=' ', mode='a')



end_time = time.time()

print('Time Elapsed:',end_time-start)
