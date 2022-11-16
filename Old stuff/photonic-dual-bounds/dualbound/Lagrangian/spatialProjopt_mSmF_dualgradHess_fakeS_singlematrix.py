#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la
from .spatialProjopt_mSmF_Zops_singlematrix import Z_TT, grad_Z_TT
from .spatialProjopt_vecs_numpy import get_Tvec, get_ZTTcho_Tvec, get_ZTTcho_Tvec_gradTvec


def get_mSmF_singlematrix_separate_duals(n_S, Lags, O, Ulist, Plist, S, ZTS_Sfunc, dualconstlist = None, include=None):
    """
    calculates the optimal multisource aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing multisource results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in O
    """
    if include is None:
        include = [True]*len(Lags)
    
    Lags_indep = Lags.copy() #Lags_indep has all cross constraint multipliers set to 0 which will give block diagonal ZTT and ZTS with each block corresponding to one source
    cross_flag = [True] * len(Lags)
    
    n_cplx_projLags = n_S**2
    for l in range(len(Plist)):
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            ind_re = ind_offset_Lags + i*n_S + i
            ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + i
            cross_flag[ind_re] = cross_flag[ind_im] = False
    Lags_indep[cross_flag] = 0.0
    
    ZTS_S_msource = ZTS_Sfunc(n_S, Lags, S)
    ZTS_S_indep = ZTS_Sfunc(n_S, Lags_indep, S)

    n_basis = Plist[0].shape[0]
    ZTT_msource = Z_TT(n_S, Lags, O, Ulist, Plist)
    T_msource = get_Tvec(ZTT_msource, ZTS_S_msource)
    
    ZTT_indep = Z_TT(n_S, Lags_indep, O, Ulist, Plist)
    duals = np.zeros(n_S)
    if dualconstlist is None:
        dualconstlist = np.zeros(n_S)
        
    for i in range(n_S):
        ZTT_i = ZTT_indep[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis]
        ZTS_S_i = ZTS_S_indep[i*n_basis:(i+1)*n_basis]
        T_i = T_msource[i*n_basis:(i+1)*n_basis]
        duals[i] = dualconstlist[i] - np.real(np.vdot(T_i, ZTT_i @ T_i)) + 2*np.real(np.vdot(T_i, ZTS_S_i))
        
    return duals


def get_mSmF_singlematrix_separate_objs(n_S, Lags, O, Ulist, Plist, S, ZTS_Sfunc, objconstlist = None, include=None):
    """
    calculates the optimal multisource aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing multisource results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in Olist
    """
    if include is None:
        include = [True]*len(Lags)
    
    Lags_obj = np.zeros(len(Lags)) #Lags_obj has all multipliers set to 0 to recover just the objective value
    
    ZTS_S_msource = ZTS_Sfunc(n_S, Lags, S)
    ZTS_S_obj = ZTS_Sfunc(n_S, Lags_obj, S)
    

    n_basis = Plist[0].shape[0]
    ZTT_msource = Z_TT(n_S, Lags, O, Ulist, Plist)
    T_msource = get_Tvec(ZTT_msource, ZTS_S_msource)
       
    ZTT_obj = Z_TT(n_S, Lags_obj, O, Ulist, Plist)

    if objconstlist is None:
        objconstlist = np.zeros(n_S)
        
    objs = np.zeros(n_S)
    for i in range(n_S):
        ZTT_i = ZTT_obj[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis]
        ZTS_S_i = ZTS_S_obj[i*n_basis:(i+1)*n_basis]
        T_i = T_msource[i*n_basis:(i+1)*n_basis]
        objs[i] = objconstlist[i] - np.real(np.vdot(T_i, ZTT_i @ T_i)) + 2*np.real(np.vdot(T_i, ZTS_S_i))
        
    return objs


def get_spatialProj_mSmF_dualgrad_fakeS_singlematrix(n_S, Lags, grad, O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = ZTS_Sfunc(n_S, Lags, S)
    
    dualval = dualconst
    if len(grad)>0:
        grad[:] = 0
        gradZTS_S = gradZTS_Sfunc(n_S, Lags, S)

        ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
        gradZTT = grad_Z_TT(n_S, Lags, Ulist, Plist)
        ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S)
        
        gradZTS_S = gradZTS_S
        
        dualval += np.real(np.vdot(T, ZTT @ T))
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
        ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
        gradZTT = grad_Z_TT(n_S, Lags, Ulist, Plist)
        ZTTcho, T = get_ZTTcho_Tvec(ZTT, ZTS_S)
        dualval += np.real(np.vdot(T, ZTT @ T))

        for _, fS in enumerate(fSlist):
            ZTTinv_fS = la.cho_solve(ZTTcho, fS)
            dualval += np.real(np.vdot(fS, ZTTinv_fS))

    return dualval


def get_spatialProj_mSmF_dualgradHess_fakeS_singlematrix(n_S, Lags, grad, Hess, O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=0.0, include=None):
    if include is None:
        include = [True]*len(Lags)
        
    ZTS_S = ZTS_Sfunc(n_S, Lags, S)
    gradZTS_S = gradZTS_Sfunc(n_S, Lags, S)
    
    grad[:] = 0
    Hess[:,:] = 0
    dualval = dualconst
    
    ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
    gradZTT = grad_Z_TT(n_S, Lags, Ulist, Plist)
    ZTTcho, T, gradT = get_ZTTcho_Tvec_gradTvec(ZTT, gradZTT, ZTS_S, gradZTS_S)
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


def get_inc_spatialProj_mSmF_dualgradHess_fakeS_singlematrix(n_S, incLags, incgrad, incHess, include, O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=0.0, get_grad=True, get_Hess=True):
    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags[:]
    
    if get_Hess:
        grad = np.zeros(Lagnum)
        Hess = np.zeros((Lagnum,Lagnum))
        dualval = get_spatialProj_mSmF_dualgradHess_fakeS_singlematrix(n_S, Lags,grad,Hess, O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include) #[:] since we are modifying in place
        incHess[:,:] = get_incHess_from_Hess(Hess, include)
    elif get_grad:
        grad = np.zeros(Lagnum)
        dualval = get_spatialProj_mSmF_dualgrad_fakeS_singlematrix(n_S, Lags,grad, O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=dualconst, include=include)
        incgrad[:] = get_incgrad_from_grad(grad, include)
    else:
        dualval = get_spatialProj_mSmF_dualgrad_fakeS_singlematrix(n_S, Lags,np.array([]), O, Ulist, Plist, S, ZTS_Sfunc, gradZTS_Sfunc, fSlist, dualconst=dualconst, include=include)
    return dualval
