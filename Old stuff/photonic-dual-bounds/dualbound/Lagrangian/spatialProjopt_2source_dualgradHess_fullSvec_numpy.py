#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:10:44 2020

@author: pengning
"""

import numpy as np
from .spatialProjopt_2source_Zops_numpy import Z_TT, grad_Z_TT
from .spatialProjopt_2source_vecs_numpy import get_Tvec, get_Tvec_gradTvec


def get_2source_separate_duals(Lags, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, dualconst1=0.0, dualconst2=0.0, include=None):
    """
    calculates the optimal 2source aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing 2 source results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in Olist
    """
    if include is None:
        include = [True]*len(Lags)
    
    Lags_indep = Lags.copy() #Lags_indep has all cross constraint multipliers set to 0 which will give block diagonal ZTT and ZTS with each block corresponding to one source
    for i in range(len(Lags)):
        imod8 = i % 8
        if imod8==1 or imod8==2 or imod8==5 or imod8==6:
            Lags_indep[i] = 0
    
    ZTS_Slist_2source = ZTS_Slistfunc(Lags, Slist)
    ZTS_Slist_indep = ZTS_Slistfunc(Lags_indep, Slist)
    
    dual1 = dualconst1
    dual2 = dualconst2
    for mode in range(len(Olist)):
        num_basis = UPlistlist[mode][0].shape[0]
        ZTT_2source = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        T_2source = get_Tvec(ZTT_2source, ZTS_Slist_2source[mode])
        
        ZTT_indep = Z_TT(Lags_indep, Olist[mode], UPlistlist[mode])
        dual1 += -np.real(np.vdot(T_2source[:num_basis], ZTT_indep[:num_basis,:num_basis] @ T_2source[:num_basis])) + 2*np.real(np.vdot(T_2source[:num_basis], ZTS_Slist_indep[mode][:num_basis]))
        dual2 += -np.real(np.vdot(T_2source[num_basis:], ZTT_indep[num_basis:,num_basis:] @ T_2source[num_basis:])) + 2*np.real(np.vdot(T_2source[num_basis:], ZTS_Slist_indep[mode][num_basis:]))
    
    return dual1, dual2


def get_2source_separate_obj(Lags, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, dualconst1=0.0, dualconst2=0.0, include=None):
    """
    calculates the optimal 2source aggregate Tvec given Lags then returns
    the dual values of each source problem independently, leaving out the cross constraints
    used for comparing 2 source results with independent 1 source results
    assumes that there is no inherent cross-coupling in the constant part of ZTT stored in Olist
    """
    if include is None:
        include = [True]*len(Lags)
    
    Lags_obj = np.zeros(len(Lags)) #Lags_obj has all multipliers set to 0 to recover just the objective value
    
    ZTS_Slist_2source = ZTS_Slistfunc(Lags, Slist)
    ZTS_Slist_obj = ZTS_Slistfunc(Lags_obj, Slist)
    
    obj1 = dualconst1
    obj2 = dualconst2
    for mode in range(len(Olist)):
        num_basis = UPlistlist[mode][0].shape[0]
        ZTT_2source = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        T_2source = get_Tvec(ZTT_2source, ZTS_Slist_2source[mode])
        
        ZTT_obj = Z_TT(Lags_obj, Olist[mode], UPlistlist[mode])
        obj1 += -np.real(np.vdot(T_2source[:num_basis], ZTT_obj[:num_basis,:num_basis] @ T_2source[:num_basis])) + 2*np.real(np.vdot(T_2source[:num_basis], ZTS_Slist_obj[mode][:num_basis]))
        obj2 += -np.real(np.vdot(T_2source[num_basis:], ZTT_obj[num_basis:,num_basis:] @ T_2source[num_basis:])) + 2*np.real(np.vdot(T_2source[num_basis:], ZTS_Slist_obj[mode][num_basis:]))
    
    return obj1, obj2


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
            T = get_Tvec(ZTT, ZTS_Slist[mode])
            #deltadual = np.real(np.vdot(T, ZTT @ T))
            #print('at mode #', mode, 'delta dual is', deltadual)
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
