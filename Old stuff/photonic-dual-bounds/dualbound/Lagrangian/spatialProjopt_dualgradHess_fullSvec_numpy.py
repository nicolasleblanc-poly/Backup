#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
from .spatialProjopt_Zops_numpy import Z_TT, grad_Z_TT
from .spatialProjopt_vecs_numpy import get_Tvec, get_Tvec_gradTvec


def get_spatialProj_dualgrad_fullS(Lags, grad, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_Slist = ZTS_Slistfunc(Lags, Slist)
    
    dualval = dualconst
    if len(grad)>0:
        grad[:] = 0
        gradZTS_Slist = gradZTS_Slistfunc(Lags, Slist)
        for mode in range(len(Olist)):
            ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
            gradZTT = grad_Z_TT(Lags, UPlistlist[mode])
            T = get_Tvec(ZTT, ZTS_Slist[mode])
            
            gradZTS_S = gradZTS_Slist[mode]
            
            dualval += np.real(np.vdot(T, ZTT @ T))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
            
    else:
        for mode in range(len(Olist)):
            ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
            gradZTT = grad_Z_TT(Lags, UPlistlist[mode])
            T = get_Tvec(ZTT, ZTS_Slist[mode])

            dualval += np.real(np.vdot(T, ZTT @ T))
            
    return dualval


def get_spatialProj_dualgradHess_fullS(Lags, grad, Hess, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_Slist = ZTS_Slistfunc(Lags, Slist)
    gradZTS_Slist = gradZTS_Slistfunc(Lags, Slist)
    
    grad[:] = 0
    Hess[:,:] = 0
    dualval = dualconst
    
    for mode in range(len(Olist)):
        ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        gradZTT = grad_Z_TT(Lags, UPlistlist[mode])
        T, gradT = get_Tvec_gradTvec(ZTT, gradZTT, ZTS_Slist[mode], gradZTS_Slist[mode])
        gradZTS_S = gradZTS_Slist[mode]
        
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


def get_inc_spatialProj_dualgradHess_fullS(incLags, incgrad, incHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_fullS(Lags,grad,Hess, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fullS(Lags,grad, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_dualgrad_fullS(Lags,np.array([]), Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
    return dualval
