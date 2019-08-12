#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:08:30 2019

@author: cbilodeau
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as col
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from keras.models import load_model

# Select Patch
patch_num = 4
slab_list = [0,24,49] #[0,4,9,14,19,24,29,34,39,44,49]

# Oneliner for overlap contours
def transparent_cmap(cmap,N=255):
    mycmap = cmap
    mycmap._init()
    mycmap._lut[:,-1] = np.linspace(0,0.8,N+4)
    return mycmap

# Set up element based colormaps: -----
greens = [(1,1,1),(0.45,0.9,0.6),(0,0.9,0.2)]
blues = [(1,1,1),(0.45,0.45,0.9),(0,0,0.9)]
reds = [(1,1,1),(0.9,0.45,0.45),(0.9,0,0)]

red1=LinearSegmentedColormap.from_list('red1',reds)
blue1=LinearSegmentedColormap.from_list('blue1',blues)
green1=LinearSegmentedColormap.from_list('green1',greens)
cmap_carbon = transparent_cmap(green1)
cmap_oxygen = transparent_cmap(red1)
cmap_nitrogen = transparent_cmap(blue1)


# Load in patch: -----
pickleFile = open('patches/train/patch'+str(patch_num)+'.pickle',"rb")
patch= pickle.load(pickleFile)

# Itentify which atom types are most prevalent: -----
for i in range(0,np.shape(patch)[0]):
    print(np.sum(patch[i]))

# Create plotting grid: -----
    
dx,dy=0.1,0.1
y,x = np.mgrid[slice(0.1,5.0+dy,dy),slice(0.1,5.0 +dx,dx)]

# Plot colormap for every slab in the patch: -----
for slab in slab_list:
    plt.figure(figsize=(1.5,1.5))
    plt.pcolor(x,y,patch[5][slab],cmap='hot')
#    plt.pcolor(x,y,patch[0][slab],cmap=cmap_carbon,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[1][slab],cmap=cmap_carbon,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[2][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[3][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[4][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[5][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch[6][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
    plt.axis('off')
    plt.show()
    plt.close

# Reformat Patch:
patch = np.swapaxes(patch,0,1)
patch = np.swapaxes(patch,1,2)
patch = np.swapaxes(patch,2,3)
patch = np.reshape(patch,[1,50,50,50,10])


# Load Encoder:
encoder = load_model('Histories/encoder.h5')

# Load Decoder:
decoder = load_model('Histories/decoder.h5')

# Plot patch encoded and then decoded:
patch_lat = encoder.predict(patch)
patch_reproduced = decoder.predict(patch_lat)
patch_reproduced = np.reshape(patch_reproduced,[50,50,50,10])
patch_reproduced = np.swapaxes(patch_reproduced,2,3)
patch_reproduced = np.swapaxes(patch_reproduced,1,2)
patch_reproduced = np.swapaxes(patch_reproduced,0,1)


dx,dy=0.1,0.1
y,x = np.mgrid[slice(0.1,5.0+dy,dy),slice(0.1,5.0 +dx,dx)]

for i in range(0,np.shape(patch_reproduced)[0]):
    print(np.sum(patch_reproduced[i]))

# Plot colormap for every slab in the patch: -----
for slab in slab_list:
    plt.figure(figsize=(1.5,1.5))
    plt.pcolor(x,y,patch_reproduced[5][slab],cmap='hot')
#    plt.pcolor(x,y,patch_reproduced[0][slab],cmap=cmap_carbon,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[1][slab],cmap=cmap_carbon,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[2][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[3][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[4][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[5][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
#    plt.pcolor(x,y,patch_reproduced[6][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
    plt.axis('off')
    plt.show()
    plt.close


