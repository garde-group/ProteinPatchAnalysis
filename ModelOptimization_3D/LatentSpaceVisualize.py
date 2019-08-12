#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:22:55 2019

@author: cbilodeau
"""

import numpy as np
import pickle
import keras
from keras.models import load_model
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as col
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap


# Settings: -----
n_training = 2000
n_validation = 900
latent_dim = 16


start = time.time()

# Open latent space representation for training data:
picklename = 'Histories/latent_rep_training.pickle'
pickle_in = open(picklename,'rb')
lat_space = pickle.load(pickle_in)
pickle_in.close()

lat_space = np.swapaxes(lat_space,0,1)
lat_space = np.reshape(lat_space,[n_training,latent_dim])

# Perform PCA on latent representation for training data:
pca = PCA(n_components=2)
pca.fit(lat_space)
lat_pca = pca.transform(lat_space)
lat_pca = np.swapaxes(lat_pca,0,1)

# Plot PCA for latent representation for training data:
plt.scatter(lat_pca[0],lat_pca[1],s=3)

# Open latent space representation for validation data:
picklename = 'Histories/latent_rep_validation.pickle'
pickle_in = open(picklename,'rb')
lat_space_val = pickle.load(pickle_in)
pickle_in.close()

lat_space_val = np.swapaxes(lat_space_val,0,1)
lat_space_val = np.reshape(lat_space_val,[n_validation,latent_dim])

# Perform PCA on latent representation for training data:
pca = PCA(n_components=2)
pca.fit(lat_space_val)
lat_pca_val = pca.transform(lat_space_val)
lat_pca_val = np.swapaxes(lat_pca_val,0,1)

# Plot PCA for latent representation for training data:
plt.scatter(lat_pca_val[0],lat_pca_val[1],s=3)

end = time.time()

print('Time Elapsed:', end-start)
