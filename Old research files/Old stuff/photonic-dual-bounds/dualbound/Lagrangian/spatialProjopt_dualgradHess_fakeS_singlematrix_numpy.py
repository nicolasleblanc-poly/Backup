#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la
from .spatialProjopt_Zops_singlematrix_numpy import get_ZTS_S, get_ZTT, get_gradZTT
from .spatialProjopt_vecs_numpy import get_Tvec, get_ZTTcho_Tvec, get_Tvec_gradTvec, get_ZTTcho_Tvec_gradTvec


def get_spatialProj_dualgrad_fakeS_singlematrix(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = get_ZTS_S(Lags, O_lin, gradZTS_S)
    ZTT = get_ZTT(Lags, O_quad, gradZTT)

    ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S)
    dualval = dualconst
    dualval += np.real(np.vdot(T, ZTT @ T))
    
    if len(grad)>0:
        grad[:] = 0
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -np.real(np.vdot(T, gradZTT[i].dot(T))) + 2*np.real(np.vdot(T, gradZTS_S[i]))

        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTinv_fS, gradZTT[i].dot(ZTTinv_fS)))

    else:
        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))

    return dualval


def get_spatialProj_dualgradHess_fakeS_singlematrix(Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = get_ZTS_S(Lags, O_lin, gradZTS_S)
    ZTT = get_ZTT(Lags, O_quad, gradZTT)
    
    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S)
    dualval = dualconst + np.real(np.vdot(T, ZTT @ T))

    grad[:] = 0
    Hess[:,:] = 0
    
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

    for _, fS in enumerate(fSlist):
        ZTTinv_fS = la.cho_solve(ZTTcho, fS)
        dualval += np.real(np.vdot(fS, ZTTinv_fS))
        ZTTinv_gradZTT_ZTTinv_fS = []
        for i in range(len(Lags)):
            if include[i]:
                gradZTT_ZTTinv_fS = gradZTT[i] @ ZTTinv_fS
                grad[i] += -np.real(np.vdot(ZTTinv_fS, gradZTT_ZTTinv_fS))
                ZTTinv_gradZTT_ZTTinv_fS.append(la.cho_solve(ZTTcho, gradZTT_ZTTinv_fS))
            else:
                ZTTinv_gradZTT_ZTTinv_fS.append(None)

        for i in range(len(Lags)):
            if not include[i]:
                continue
            for j in range(i,len(Lags)):
                if not include[j]:
                    continue
                Hess[i,j] += 2*np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_gradZTT_ZTTinv_fS[j]))
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


def get_inc_spatialProj_dualgradHess_fakeS_singlematrix(incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_fakeS_singlematrix(Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_singlematrix(Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_dualgrad_fakeS_singlematrix(Lags,np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
    return dualval
