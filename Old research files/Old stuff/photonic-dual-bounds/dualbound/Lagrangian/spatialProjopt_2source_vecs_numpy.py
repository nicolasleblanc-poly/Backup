#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  8 20:27:07 2020

@author: pengning
"""

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import warnings

def get_Tvec(ZTT, ZTS_S):
    with warnings.catch_warnings():
        warnings.filterwarnings('error')
        try:
            Tvec = sp.linalg.solve(ZTT, ZTS_S, assume_a='her')
            #Tvec = sp.linalg.solve(ZTT, ZTS_S, assume_a='pos')
        except Warning as e:
            print('error found:', e)
            raise FloatingPointError('ill conditioned matrix encountered in Tsolve')
    return Tvec


def get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    #print('sum over ZTT elements', np.sum(ZTT))
    #print('colormap of ZTT')
    #plt.figure()
    #plt.imshow(np.abs(ZTT))
    #plt.show()
    #print('ZTS_S', ZTS_S)
    ZTT_lu, ZTT_piv = sp.linalg.lu_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    Tvec = sp.linalg.lu_solve((ZTT_lu,ZTT_piv), ZTS_S)
    #print('Tvec', Tvec)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(sp.linalg.lu_solve((ZTT_lu,ZTT_piv), -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return Tvec, gradTvec


"""
def get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    
    ZTT_chofac = sp.linalg.cho_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    
    Tvec = sp.linalg.cho_solve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(sp.linalg.cho_solve(ZTT_chofac, -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return Tvec, gradTvec
"""