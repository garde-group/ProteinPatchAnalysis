#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 16:04:14 2019

@author: cbilodeau
"""

import numpy as np
import pickle
import keras
from keras.models import load_model
import time

# Settings: -----
n_training = 2000
n_validation = 900
lat_space = []

start = time.time()

# Load encoder: -----
encoder = load_model('Histories/encoder.h5')



for i in range(0,n_training):
    print('Adding patch',i)
    # Load in Patch:
    patch_name = 'patch'+str(i)
    patch_path = 'patches/train/patch' + patch_name + '.pickle'
    pickle_in = open('patches/train/patch'+str(i)+'.pickle','rb')
    patch = pickle.load(pickle_in)
    pickle_in.close()
    
    # Reformat Patch:
    patch = np.swapaxes(patch,0,1)
    patch = np.swapaxes(patch,1,2)
    patch = np.swapaxes(patch,2,3)
    patch = np.reshape(patch,[1,50,50,50,10])
    
    # Obtain Latent Representation:
    lat_rep = encoder.predict(patch)
    
    # Compile Latent Space:
    lat_space.append(lat_rep)
    
picklename = 'Histories/latent_rep_training.pickle'
pickle_out = open(picklename,'wb')
pickle.dump(lat_space,pickle_out)
pickle_out.close()

lat_space = []

for i in range(n_training,n_training+n_validation):
    print('Adding patch',i)
    # Load in Patch:
    patch_name = 'patch'+str(i)
    patch_path = 'patches/validation/patch' + patch_name + '.pickle'
    pickle_in = open('patches/validation/patch'+str(i)+'.pickle','rb')
    patch = pickle.load(pickle_in)
    pickle_in.close()
    
    # Reformat Patch:
    patch = np.swapaxes(patch,0,1)
    patch = np.swapaxes(patch,1,2)
    patch = np.swapaxes(patch,2,3)
    patch = np.reshape(patch,[1,50,50,50,10])
    
    # Obtain Latent Representation:
    lat_rep = encoder.predict(patch)
    
    # Compile Latent Space:
    lat_space.append(lat_rep)
    
picklename = 'Histories/latent_rep_validation.pickle'
pickle_out = open(picklename,'wb')
pickle.dump(lat_space,pickle_out)
pickle_out.close()


end = time.time()

print('Time Elapsed:', end-start)
