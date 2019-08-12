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

# Settings: -----
n_training = 2000


# Load encoder: -----
encoder = load_model('encoder.h5')



for i in range(0,n_training):
    patch_name = 'patch'+str(i)
    patch_path = 'patches/train/patch' + patch_name + '.pickle'
    pickle_in = open('patches/train/patch'+str(i)+'.pickle','rb')
    patch = pickle.load(pickle_in)
    pickle_in.close()
    patch =  np.swapaxes(patch,0,1)
    patch =  np.swapaxes(patch,1,2)
    patch =  np.swapaxes(patch,2,3)
    lat_rep = encoder.predict(patch)