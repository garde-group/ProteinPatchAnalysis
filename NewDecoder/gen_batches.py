#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 11:13:55 2019

Function to generate batches for vae training
This is necessary because it reduces the amount of data that needs to be held
in memory while training

@author: cbilodeau
"""

import numpy as np
import os
import pickle

def gen_batches(batch_size=10):
    while True:


        patch_path = os.getcwd()+'/patches/train'
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
        
        
        
def gen_batches_validation(batch_size=10):
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

    
#    patch_path = os.getcwd()+'/patches'
#    all_patches = os.listdir(patch_path)
#    
#    # Shuffle patch list:
#    random.shuffle(all_patches)
#    
#    size = len(all_patches) // batch_size
#    leftovers = all_patches[size*batch_size:]
#    
#    # Create Batch List:
#    batch_list = []
#    for i in range(0,size):
#        batch_list.append(all_patches[i*batch_size:(i*batch_size+batch_size-1)])
#        
#    batch_list.append(leftovers)    
#    
#    batch_gen = (batch for batch in batch_list)
#    
#    return batch_gen
#

    