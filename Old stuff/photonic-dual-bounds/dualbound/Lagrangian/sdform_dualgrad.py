#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
from .spatialProjopt_vecs_numpy import get_Tvec


def get_sdform_dualgrad(Lags, grad, O_lin_eta, O_quad_eta, gradLzy_y, gradLzz, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)

    Lagnum = len(Lags)
    Lzy_y = O_lin_eta.copy()
    Lzz = O_quad_eta.copy()
    for i in range(Lagnum):
        if include[i]:
            Lzy_y += Lags[i]*gradLzy_y[i]
            Lzz += Lags[i]*gradLzz[i]

    z = get_Tvec(Lzz, Lzy_y)
    
    dualval = dualconst + np.real(np.vdot(z, Lzz @ z))
    
    if len(grad)>0:
        grad[:] = 0
        for i in range(Lagnum):
            if include[i]:
                grad[i] += -np.real(np.vdot(z, gradLzz[i]@z)) + 2*np.real(np.vdot(z, gradLzy_y[i]))
        
    return dualval


def get_inc_sdform_dualgrad(incLags, incgrad, include, O_lin_eta, O_quad_eta, gradLzy_y, gradLzz, dualconst=0.0):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if len(incgrad)>0:
        grad = np.zeros(Lagnum)
        dualval = get_sdform_dualgrad(Lags, grad, O_lin_eta, O_quad_eta, gradLzy_y, gradLzz, dualconst=dualconst, include=include)
        incgrad[:] = grad[include]
    else:
        dualval = get_sdform_dualgrad(Lags, np.array([]), O_lin_eta, O_quad_eta, gradLzy_y, gradLzz, dualconst=dualconst, include=include)
    
    return dualval
