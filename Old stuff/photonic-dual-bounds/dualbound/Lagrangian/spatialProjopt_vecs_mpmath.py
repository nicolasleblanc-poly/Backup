#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 23:02:18 2020

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
from .mpmatrix_tools import mp_CholeskyLsolve

def get_Tvec(ZTT, ZTS_S):
    Tvec = mp.lu_solve(ZTT, ZTS_S)
    return Tvec


def get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S):
    
    ZTT_chofac = mp.cholesky(ZTT) #later on need many solves with ZTT as coeff matrix, so do decomposition beforehand
    
    Tvec = mp_CholeskyLsolve(ZTT_chofac, ZTS_S)
    
    gradTvec = []
    for i in range(len(gradZTT)):
        gradTvec.append(mp_CholeskyLsolve(ZTT_chofac, -gradZTT[i] * Tvec + gradZTS_S[i]))
    
    return Tvec, gradTvec
