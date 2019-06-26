#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:03:59 2019

Function to generate batches for vae training on VALIDATION DATA
This is necessary because it reduces the amount of data that needs to be held
in memory while trainin

@author: cbilodeau
"""

import numpy as np
import os
import pickle

def gen_batches(batch_size=10):
    while True:


        patch_path = os.getcwd()+'/patches/validation'
        all_patches = os.listdir(patch_path)  
        batch_paths = np.random.choice(all_patches,size=batch_size)
        batch_input=[]

        for input_path in batch_paths:
            pickle_in = open(patch_path+"/"+input_path,"rb")
            input = pickle.load(pickle_in)
            
            input = np.swapaxes(input,0,1)
            input = np.swapaxes(input,1,2)
            input = np.swapaxes(input,2,3)

            #print(np.shape(input))
            
            batch_input += [ input ]
        
        batch_input = np.array(batch_input)
#        print(batch_input)
#            batch_input=1
    
        yield(batch_input,None)
