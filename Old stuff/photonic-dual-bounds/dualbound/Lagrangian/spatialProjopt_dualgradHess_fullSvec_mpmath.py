#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
from .mpmatrix_tools import mp_conjdot
from .spatialProjopt_Zops_mpmath import Z_TT, grad_Z_TT
from .spatialProjopt_vecs_mpmath import get_Tvec, get_Tvec_gradTvec


def get_spatialProj_dualgrad_fullS(Lags, grad, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=mp.zero, include=None):
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
            
            dualval += mp.re(mp_conjdot(T, ZTT * T))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -mp.re(mp_conjdot(T, gradZTT[i]*T)) + 2*mp.re(mp_conjdot(T, gradZTS_S[i]))
    else:
        for mode in range(len(Olist)):
            ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
            gradZTT = grad_Z_TT(Lags, UPlistlist[mode])
            T = get_Tvec(ZTT, ZTS_Slist[mode])

            dualval += mp.re(mp_conjdot(T, ZTT * T))
            
    return dualval


def get_spatialProj_dualgradHess_fullS(Lags, grad, Hess, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=mp.zero, include=None):
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
        
        dualval += mp.re(mp_conjdot(T, ZTT * T))
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -mp.re(mp_conjdot(T, gradZTT[i]*T)) + 2*mp.re(mp_conjdot(T, gradZTS_S[i]))
        for i in range(len(Lags)):
            if not include[i]:
                continue
            for j in range(i,len(Lags)):
                if not include[j]:
                    continue
                Hess[i,j] += 2*mp.re(mp_conjdot(gradT[i], -gradZTT[j]*T + gradZTS_S[j]))
                if i!=j:
                    Hess[j,i] = Hess[i,j]
    
    return dualval


def get_Lags_from_incLags(incLags, include):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum, dtype=type(mp.one))
    Lags[include] = np.squeeze(incLags.tolist())
    return mp.matrix(Lags)

def get_incgrad_from_grad(grad, include):
    return mp.matrix(np.squeeze(grad.tolist())[include])

def get_incHess_from_Hess(Hess, include):
    return mp.matrix(np.array(Hess.tolist())[np.ix_(include,include)])


def get_inc_spatialProj_dualgradHess_fullS(incLags, incgrad, incHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=mp.zero, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = get_Lags_from_incLags(incLags, include)
    
    if get_Hess:
        grad = mp.zeros(Lagnum,1)
        Hess = mp.zeros(Lagnum,Lagnum)
        dualval = get_spatialProj_dualgradHess_fullS(Lags,grad,Hess, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = mp.zeros(Lagnum,1)
        dualval = get_spatialProj_dualgrad_fullS(Lags,grad, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_dualgrad_fullS(Lags,[], Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, dualconst=dualconst, include=include)
    return dualval
