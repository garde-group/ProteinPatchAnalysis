#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 10:19:29 2019

@author: cbilodeau
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt

# Settings from training:
n_epochs_range = [100]
batch_size_range = [256] #[32,64,128,256,512,1024]
learning_rate_range = [0.0075, 0.01] #[0.001,0.0025,0.005,0.0075,0.01]
decay_rate_range = [0.1,0.2] #,0.3]
latent_dim_range = [2]

# Initialize Variables:
stats = {}
name_list = []
time_avgs = []
time_stds = []
test_loss_compiled = []
val_loss_compiled = []
val_avgs = []
val_stds = []
test_fin = []

# Graphing Settings:
epoch_skip = 50

for n_epochs in n_epochs_range:
    for batch_size in batch_size_range:
        for learning_rate in learning_rate_range:
            for decay_rate in decay_rate_range:
                for latent_dim in latent_dim_range:
                    name = str(batch_size)+'-'+str(learning_rate)+'-'+str(decay_rate)
                    in_name='stats'+'-'+name+'.pickle'
                    pickleFile = open(in_name,"rb")
                    stats[name]= pickle.load(pickleFile)
                    
                    name_list.append(name)
                    
                    time_avgs.append(np.mean(stats[name]['epoch_time']))
                    time_stds.append(np.std(stats[name]['epoch_time'][1:]))
                    
                    test_loss_compiled.append(stats[name]['test_loss'])
                    val_loss_compiled.append(stats[name]['val_loss'])
                    
                    val_avgs.append(np.mean(stats[name]['val_loss'][epoch_skip:]))
                    val_stds.append(np.std(stats[name]['val_loss'][epoch_skip:][1:]))
                    
                    test_fin.append(stats[name]['test_loss'][-1])
                    

# Plot Time Statistics -------------------------
plt.bar(name_list,time_avgs,yerr = time_stds)
plt.show()
plt.close()

# Plot All Test Loss Curves --------------------
i=0
for run in test_loss_compiled:
    plt.plot(run[3:],label=name_list[i])
    i+=1
plt.legend()
plt.show()
plt.close()


# Plot All Validation Loss Curves ---------------
i=0
for run in val_loss_compiled:
    plt.plot(run[2:],label=name_list[i])
    i+=1
plt.legend()
plt.show()
plt.close()

# Plot Validation Loss Statistics ----------------
plt.bar(name_list,val_avgs,yerr = val_stds)
plt.bar(name_list,test_fin)
plt.show()
plt.close()


#pickleFile = open("stats-32-0.001-0.0.pickle","rb")
#stats32 = pickle.load(pickleFile)
#
#pickleFile = open("stats-64-0.001-0.0.pickle","rb")
#stats64 = pickle.load(pickleFile)
#
#pickleFile = open("stats-128-0.001-0.0.pickle","rb")
#stats128 = pickle.load(pickleFile)
#
#pickleFile = open("stats-256-0.001-0.0.pickle","rb")
#stats256 = pickle.load(pickleFile)
#
#
## Time Comparisons:
#avg_times = 
#
#plt.bar(0,np.mean(stats32['epoch_time'][1:]))
##plt.plot(stats32['epoch_time'][1:])
##plt.plot(stats64['epoch_time'][1:])
##plt.plot(stats128['epoch_time'][1:])
##plt.plot(stats256['epoch_time'][1:])
#plt.show()

#plt.plot(stats32['test_loss'][1:])
#plt.plot(stats32['val_loss'][1:])
#plt.show()
#plt.close()
#
#plt.plot(stats64['test_loss'][1:])
#plt.plot(stats64['val_loss'][1:])
#plt.show()
#plt.close()
#
#plt.plot(stats128['test_loss'][1:])
#plt.plot(stats128['val_loss'][1:])
#plt.show()
#plt.close()
#
#plt.plot(stats256['test_loss'][1:])
#plt.plot(stats256['val_loss'][1:])
#plt.show()
#plt.close()
#
#plt.plot(stats256['test_loss'][1:])
#plt.show()
#plt.close()
