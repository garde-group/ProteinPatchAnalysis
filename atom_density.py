# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 17:15:14 2019

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

# Calculate density felt at a point from a distance from an atom center:
def atom_density(distance,std):
    density = np.exp(-distance**2/(2*std**2))
    return density