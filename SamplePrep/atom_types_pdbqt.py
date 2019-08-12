# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 10:10:18 2019

@author: camil
"""

import numpy as np

def atom_id_pdbqt(name):
    n_atomtypes = 10
    id_mat = np.zeros([1,n_atomtypes])[0]
    
    # Atom type 1: Carbon
    if name=='C':
        id_mat[0] = 1

    # Atom type 2: Carbon A
    if name=='A':
        id_mat[1] = 1
 
    # Atom type 3: Nitrogen
    if name=='N':
        id_mat[2] = 1

    # Atom type 4: Nitrogen A
    if name=='NA':
        id_mat[3] = 1

    # Atom type 5: Nitrogen S
    if name=='NS':
        id_mat[4] = 1

    # Atom type 6: Oxygen A
    if name=='OA':
        id_mat[5] = 1

    # Atom type 7: Oxygen A
    if name=='OS':
        id_mat[6] = 1

    # Atom type 8: Oxygen A
    if name=='SA':
        id_mat[7] = 1
        
    # Atom type 9: Oxygen A
    if name=='HD':
        id_mat[8] = 1

    # Atom type 10: Oxygen A
    if name=='HS':
        id_mat[9] = 1

#    # Atom type 11: Oxygen A
#    if name=='MG':
#        id_mat[10] = 1
#
#    # Atom type 12: Oxygen A
#    if name=='ZN':
#        id_mat[11] = 1
#
#    # Atom type 13: Oxygen A
#    if name=='MN':
#        id_mat[12] = 1
#
#    # Atom type 14: Oxygen A
#    if name=='CA':
#        id_mat[13] = 1
#        
#     # Atom type 15: Oxygen A
#    if name=='FE':
#        id_mat[14] = 1
 
    return id_mat
