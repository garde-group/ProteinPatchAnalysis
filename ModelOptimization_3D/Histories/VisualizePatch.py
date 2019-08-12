# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 16:40:59 2019

@author: camil
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as col
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

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
pickleFile = open('patch0.pickle',"rb")
patch= pickle.load(pickleFile)

# Itentify which atom types are most prevalent: -----
for i in range(0,np.shape(patch)[0]):
    print(np.sum(patch[i]))

# Create plotting grid: -----


    
dx,dy=0.1,0.1
y,x = np.mgrid[slice(0.1,5.0+dy,dy),slice(0.1,5.0 +dx,dx)]
#fig = plt.figure(figsize=(2,2))
#gs = gridspec.GridSpec(1,1)
#ax1 = fig.add_subplot(gs[0,0])

# Plot colormap for every slab in the patch: -----
for slab in [0,4,9,14,19,24,29,34,39,44,49]:
    plt.figure(figsize=(1.5,1.5))
    plt.pcolor(x,y,patch[0][slab],cmap=cmap_carbon,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[1][slab],cmap=cmap_carbon,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[2][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[3][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[4][slab],cmap=cmap_nitrogen,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[5][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
    plt.pcolor(x,y,patch[6][slab],cmap=cmap_oxygen,vmin=0,vmax=1)
    plt.axis('off')
    plt.show()
    plt.close



#
#for slab in [0,9,19,29,39,49]:
#    plt.imshow(patch[0][slab])
#    plt.show()
#    plt.close
#    
