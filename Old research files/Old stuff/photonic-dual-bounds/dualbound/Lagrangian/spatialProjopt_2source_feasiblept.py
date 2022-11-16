#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:48:20 2020

@author: pengning
"""

import numpy as np
import scipy.optimize as sopt
from .spatialProjopt_2source_Zops_numpy import Z_TT, grad_Z_TT, get_ZTT_mineig_grad

feasiblept = None

def Lags_normsqr(Lags):
    return np.sum(Lags*Lags), 2*Lags

def Lags_normsqr_Hess_np(Lags):
    return 2*np.eye(len(Lags))

def get_modes_ZTT_mineig(incLags, include, Olist, UPlistlist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    mineiglist = np.zeros(len(Olist))
    for mode in range(len(Olist)):
        ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        eigw = np.linalg.eigvalsh(ZTT)
        mineiglist[mode] = eigw[0]
    
    if np.min(mineiglist)>=0:
        global feasiblept
        print('mineigs for all primalHess')
        print(mineiglist)
        feasiblept = incLags
        raise ValueError('found a feasible point')
    return mineiglist

def get_modes_ZTT_gradmineig(incLags, include, Olist, UPlistlist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    mineigJac = np.zeros((len(Olist),len(incLags)))
    for mode in range(len(Olist)):
        ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        grad_ZTT = grad_Z_TT(Lags, UPlistlist[mode])
        mineigJac[mode,:] = get_ZTT_mineig_grad(ZTT, grad_ZTT)[include]
    return mineigJac

def spatialProjopt_2source_find_feasiblept(Lagnum, include, Olist, UPlistlist):
    incLagnum = np.sum(include)
    initincLags = np.random.rand(incLagnum)
    
    mineigincfunc = lambda incL: get_modes_ZTT_mineig(incL, include, Olist, UPlistlist)
    Jacmineigincfunc = lambda incL: get_modes_ZTT_gradmineig(incL, include, Olist, UPlistlist)
    
    tolcstrt = 1e-4
    cstrt = sopt.NonlinearConstraint(mineigincfunc, tolcstrt, np.inf, jac=Jacmineigincfunc, keep_feasible=False)
    
    lb = -np.inf*np.ones(incLagnum)
    ub = np.inf*np.ones(incLagnum)
    bnds = sopt.Bounds(lb,ub)
    
    try:
        res = sopt.minimize(Lags_normsqr, initincLags, method='trust-constr', jac=True, hess=Lags_normsqr_Hess_np,
                            bounds=bnds, constraints=cstrt, options={'verbose':2,'maxiter':300})
    except ValueError:
        global feasiblept
        Lags = np.zeros(Lagnum)
        Lags[include] = feasiblept
        return Lags
    
    Lags = np.zeros(Lagnum)
    Lags[include] = res.x
    Lags[1] = np.abs(Lags[1]) + 0.01
    return Lags

