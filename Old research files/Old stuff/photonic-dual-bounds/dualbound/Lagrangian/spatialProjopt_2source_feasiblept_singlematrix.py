#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:48:20 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as sopt
from .spatialProjopt_2source_Zops_singlematrix_numpy import Z_TT, grad_Z_TT, get_ZTT_mineig_grad

feasiblept = None

def Lags_normsqr(Lags):
    return np.sum(Lags*Lags), 2*Lags

def Lags_normsqr_Hess_np(Lags):
    return 2*np.eye(len(Lags))

def get_ZTT_mineig(incLags, include, O, UPlist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    ZTT = Z_TT(Lags, O, UPlist)
    eigw = la.eigvalsh(ZTT)
    if eigw[0]>=0:
        global feasiblept
        feasiblept = incLags
        raise ValueError('found a feasible point')
    return eigw[0]


def get_ZTT_gradmineig(incLags, include, O, UPlist):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    ZTT = Z_TT(Lags, O, UPlist)
    grad_ZTT = grad_Z_TT(Lags, UPlist)
    mineigJac = np.zeros((1,len(incLags)))
    mineigJac[0,:] = get_ZTT_mineig_grad(ZTT, grad_ZTT)[include]
    return mineigJac


def spatialProjopt_find_feasiblept(Lagnum, include, O, UPlist):
    incLagnum = np.sum(include)
    initincLags = np.random.rand(incLagnum)
    
    mineigincfunc = lambda incL: get_ZTT_mineig(incL, include, O, UPlist)
    Jacmineigincfunc = lambda incL: get_ZTT_gradmineig(incL, include, O, UPlist)
    
    tolcstrt = 1e-4
    cstrt = sopt.NonlinearConstraint(mineigincfunc, tolcstrt, np.inf, jac=Jacmineigincfunc, keep_feasible=False)
    
    lb = -np.inf*np.ones(incLagnum)
    ub = np.inf*np.ones(incLagnum)
    bnds = sopt.Bounds(lb,ub)
    
    try:
        res = sopt.minimize(Lags_normsqr, initincLags, method='trust-constr', jac=True, hess=Lags_normsqr_Hess_np,
                            bounds=bnds, constraints=cstrt, options={'verbose':2,'maxiter':100})
    except ValueError:
        global feasiblept
        Lags = np.zeros(Lagnum)
        Lags[include] = feasiblept
        return Lags
    
    Lags = np.zeros(Lagnum)
    Lags[include] = res.x
    Lags[1] = np.abs(Lags[1]) + 0.01
    return Lags

