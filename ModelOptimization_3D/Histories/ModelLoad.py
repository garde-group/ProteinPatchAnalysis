#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 14:02:48 2019

@author: cbilodeau
"""
import keras
import numpy as np
from keras.models import load_model
import pickle
#from keras import load_model
#from keras.models import model_from_json


## Model reconstruction from JSON file
#with open('model_architecture.json', 'r') as f:
#    model = model_from_json(f.read())
#
## Load weights into the new model
#model.load_weights('model_weights.h5')

encoder = load_model('encoder.h5')
decoder = load_model('decoder.h5')

pickle_in = open('..//patches/train/patch0.pickle','rb')
patch = pickle.load(pickle_in)
patch =  np.swapaxes(patch,0,1)
patch =  np.swapaxes(patch,1,2)
patch =  np.swapaxes(patch,2,3)
