#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
from .spatialProjopt_Zops_singlematrix_numpy import Z_TT, grad_Z_TT
from .spatialProjopt_vecs_numpy import get_Tvec, get_Tvec_gradTvec


def get_spatialProj_dualgrad_singlematrix(Lags, grad, O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = ZTS_Sfunc(Lags, S)
    
    dualval = dualconst
    if len(grad)>0:
        grad[:] = 0
        gradZTS_S = gradZTS_Sfunc(Lags, S)

        ZTT = Z_TT(Lags, O, UPlist)
        gradZTT = grad_Z_TT(Lags, UPlist)
        T = get_Tvec(ZTT, ZTS_S)
            
        gradZTS_S = gradZTS_S
        
        dualval += np.real(np.vdot(T, ZTT @ T))
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
            
    else:
        ZTT = Z_TT(Lags, O, UPlist)
        gradZTT = grad_Z_TT(Lags, UPlist)
        T = get_Tvec(ZTT, ZTS_S)
        dualval += np.real(np.vdot(T, ZTT @ T))

    return dualval


def get_spatialProj_dualgradHess_singlematrix(Lags, grad, Hess, O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = ZTS_Sfunc(Lags, S)
    gradZTS_S = gradZTS_Sfunc(Lags, S)
    
    grad[:] = 0
    Hess[:,:] = 0
    dualval = dualconst
    
    ZTT = Z_TT(Lags, O, UPlist)
    gradZTT = grad_Z_TT(Lags, UPlist)
    T, gradT = get_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S)
    gradZTS_S = gradZTS_S
        
    dualval += np.real(np.vdot(T, ZTT @ T))
    for i in range(len(Lags)):
        if include[i]:
            grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
            
    for i in range(len(Lags)):
        if not include[i]:
            continue
        for j in range(i,len(Lags)):
            if not include[j]:
                continue
            Hess[i,j] += 2*np.real(np.vdot(gradT[i],-gradZTT[j] @ T + gradZTS_S[j]))
            if i!=j:
                Hess[j,i] = Hess[i,j]

    return dualval


def get_Lags_from_incLags(incLags, include):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags
    return Lags

def get_incgrad_from_grad(grad, include):
    return grad[include]

def get_incHess_from_Hess(Hess, include):
    return Hess[np.ix_(include,include)]


def get_inc_spatialProj_dualgradHess_singlematrix(incLags, incgrad, incHess, include, O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_singlematrix(Lags,grad,Hess, O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_singlematrix(Lags,grad, O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_dualgrad_singlematrix(Lags,np.array([]), O, UPlist, S, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include)
    return dualval
