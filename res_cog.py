# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:09:23 2019

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

from Bio.PDB.PDBParser import PDBParser
parser = PDBParser(PERMISSIVE=1)

# Calculate center of geometry for a given residue:
def res_cog(residue):
    coord = [residue.get_list()[i].get_coord() for i in range(0,np.shape(residue.get_list())[0])]
    cog = np.mean(coord,axis=0)
    return cog