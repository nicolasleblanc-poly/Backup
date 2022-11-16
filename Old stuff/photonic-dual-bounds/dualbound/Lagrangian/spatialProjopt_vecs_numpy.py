#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:02:18 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la


def get_Tvec(ZTT, ZTS_S):
    Tvec = la.solve(ZTT, ZTS_S, assume_a='her')
    #Tvec = la.solve(ZTT, ZTS_S, assume_a='pos')
    return Tvec

def get_ZTTcho_Tvec(ZTT, ZTS_S):
    ZTT_chofac = la.cho_factor(ZTT)
    Tvec = la.cho_solve(ZTT_chofac, ZTS_S)
    return ZTT_chofac, Tvec


def get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    
    ZTT_chofac = la.cho_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    
    Tvec = la.cho_solve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(la.cho_solve(ZTT_chofac, -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return Tvec, gradTvec


def get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    """
    exactly the same as method above with additional return of ZTT_chofac for fakeS term computations
    """
    
    ZTT_chofac = la.cho_factor(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    
    Tvec = la.cho_solve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(la.cho_solve(ZTT_chofac, -gradZTT[i] @ Tvec + gradZTS_S[i]))
    
    return ZTT_chofac, Tvec, gradTvec
