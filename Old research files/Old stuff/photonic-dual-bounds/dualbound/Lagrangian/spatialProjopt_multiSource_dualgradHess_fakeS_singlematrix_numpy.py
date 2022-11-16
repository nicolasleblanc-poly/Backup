#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la
from .spatialProjopt_multiSource_Zops_singlematrix_numpy import get_ZTT, get_gradZTT
from .spatialProjopt_vecs_numpy import get_Tvec, get_ZTTcho_Tvec, get_ZTTcho_Tvec_gradTvec


def get_multiSource_singlematrix_separate_duals(n_S, Lags, O_lin, O_quad, gradZTS_S, gradZTT, dualconstlist = None, include=None):
    """
    calculates the optimal multisource aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing multisource results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in O
    """
    if include is None:
        include = [True]*len(Lags)

    n_basis = len(O_lin) // n_S
    
    Lags_indep = Lags.copy() #Lags_indep has all cross constraint multipliers set to 0 which will give block diagonal ZTT and ZTS with each block corresponding to one source
    cross_flag = [True] * len(Lags)
    
    n_cplx_projLags = n_S**2
    n_proj = len(Lags)//(2*n_cplx_projLags)
    for l in range(n_proj):
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            ind_re = ind_offset_Lags + i*n_S + i
            ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + i
            cross_flag[ind_re] = cross_flag[ind_im] = False
    Lags_indep[cross_flag] = 0.0
    
    ZTS_S_msource = O_lin.copy()
    ZTS_S_indep = O_lin.copy()
    for i in range(len(gradZTS_S)):
        ZTS_S_msource += Lags[i] * gradZTS_S[i]
        ZTS_S_indep += Lags_indep[i] * gradZTS_S[i]


    ZTT_msource = O_quad.copy()
    ZTT_indep = O_quad.copy()
    for i in range(len(gradZTT)):
        ZTT_msource += Lags[i] * gradZTT[i]
        ZTT_indep += Lags_indep[i] * gradZTT[i]
        
    T_msource = get_Tvec(ZTT_msource, ZTS_S_msource)
    
    duals = np.zeros(n_S)
    if dualconstlist is None:
        dualconstlist = np.zeros(n_S)
        
    for i in range(n_S):
        ZTT_i = ZTT_indep[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis]
        ZTS_S_i = ZTS_S_indep[i*n_basis:(i+1)*n_basis]
        T_i = T_msource[i*n_basis:(i+1)*n_basis]
        duals[i] = dualconstlist[i] - np.real(np.vdot(T_i, ZTT_i @ T_i)) + 2*np.real(np.vdot(T_i, ZTS_S_i))
        
    return duals


def get_multiSource_singlematrix_separate_objs(n_S, Lags, O_lin, O_quad, gradZTS_S, gradZTT, objconstlist = None, include=None):
    """
    calculates the optimal multisource aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing multisource results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in Olist
    """
    if include is None:
        include = [True]*len(Lags)

    n_basis = len(O_lin) // n_S
    
    Lags_obj = np.zeros(len(Lags)) #Lags_obj has all multipliers set to 0 to recover just the objective value
    
    ZTS_S_msource = O_lin.copy()
    ZTS_S_obj = O_lin.copy()
    for i in range(len(gradZTS_S)):
        ZTS_S_msource += Lags[i] * gradZTS_S[i]
        ZTS_S_obj += Lags_obj[i] * gradZTS_S[i]


    ZTT_msource = O_quad.copy()
    ZTT_obj = O_quad.copy()
    for i in range(len(gradZTT)):
        ZTT_msource += Lags[i] * gradZTT[i]
        ZTT_obj += Lags_obj[i] * gradZTT[i]

    T_msource = get_Tvec(ZTT_msource, ZTS_S_msource)
    
    if objconstlist is None:
        objconstlist = np.zeros(n_S)
        
    objs = np.zeros(n_S)
    for i in range(n_S):
        ZTT_i = ZTT_obj[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis]
        ZTS_S_i = ZTS_S_obj[i*n_basis:(i+1)*n_basis]
        T_i = T_msource[i*n_basis:(i+1)*n_basis]
        objs[i] = objconstlist[i] - np.real(np.vdot(T_i, ZTT_i @ T_i)) + 2*np.real(np.vdot(T_i, ZTS_S_i))
        
    return objs


def get_spatialProj_dualgrad_fakeS_singlematrix(n_S, Lags, grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)

    ZTT = O_quad.copy()
    ZTS_S = O_lin.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
        ZTS_S += Lags[i] * gradZTS_S[i]

    ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S)
    dualval = dualconst + np.real(np.vdot(T, ZTT @ T))
    if len(grad)>0:
        grad[:] = 0
    
        for i in range(len(Lags)):
            if include[i]:
                grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))

        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))
            for i in range(len(Lags)):
                if include[i]:
                    grad[i] += -np.real(np.vdot(ZTTinv_fS, gradZTT[i] @ ZTTinv_fS))

    else:

        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))

    return dualval


def get_spatialProj_dualgradHess_fakeS_singlematrix(n_S, Lags, grad, Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTT = O_quad.copy()
    ZTS_S = O_lin.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
        ZTS_S += Lags[i] * gradZTS_S[i]

    
    grad[:] = 0
    Hess[:,:] = 0
    dualval = dualconst
    
    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S)
        
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


def get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(n_S, incLags, incgrad, incHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_dualgradHess_fakeS_singlematrix(n_S, Lags,grad,Hess, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_dualgrad_fakeS_singlematrix(n_S, Lags,grad, O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_dualgrad_fakeS_singlematrix(n_S, Lags,np.array([]), O_lin, O_quad, gradZTS_S, gradZTT, fSlist, dualconst=dualconst, include=include)
    return dualval
