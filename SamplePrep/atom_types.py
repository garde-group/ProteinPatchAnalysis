#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 11 11:48:29 2019

Function that takes in atom object and outputs array containing atom identity
To add atom types, just create a condition and add an atom type. Note that it
will be necessary to increase n_atomtypes in VoxelParse.py

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
from math import factorial
import random
import time
import Bio
import pickle

from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)

def atom_id(atom):
    n_atomtypes = 4
    id_mat = np.zeros([1,n_atomtypes])[0]
    # Atom type 1: Carbon
    if atom.get_name()[0]=='C':
        id_mat[0] = 1
        
    # Atom type 2: Nitrogen
    if atom.get_name()[0]=='N':
        id_mat[1] = 1
        
    # Atom type 3: Oxygen
    if atom.get_name()[0]=='O':
        id_mat[2] = 1
    
    # Atom type 4: Sulfur
    if atom.get_name()[0]=='S':
        id_mat[3] = 1
    
    return id_mat

