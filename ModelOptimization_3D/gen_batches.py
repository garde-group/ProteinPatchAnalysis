#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:55 2019

Contains functions to generate batches for vae training
This is necessary because it reduces the amount of data that needs to be held
in memory while training

@author: cbilodeau
"""

import numpy as np
import os
import pickle

# Generate batches for training data:
def gen_batches(batch_size=10):
    while True:

        # Setup list of size batch_size containing paths to
        # patches:

        patch_path = os.getcwd()+'/patches/train'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        batch_input=[]

        for input_path in batch_paths:
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)
            
            # Swap axes to create channels last format
            # (adjust in prep script later)
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)
            
            batch_input += [ input ]
        
        batch_input = np.array(batch_input)
    
        yield(batch_input,None)
        
        
# Generate batches for validation data:
        
def gen_batches_validation(batch_size=10):
    while True:

        # Setup list of size batch_size containing paths to
        # patches:

        patch_path = os.getcwd()+'/patches/validation'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        batch_input=[]

        for input_path in batch_paths:
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)

            # Swap axes to create channels last format
            # (adjust in prep script later)
            
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)

            
            batch_input += [ input ]
        
        batch_input = np.array(batch_input)
    
        yield(batch_input,None)
