#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug  9 16:48:20 2020

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import scipy.optimize as sopt
from .spatialProjopt_multiSource_Zops_singlematrix_numpy import get_ZTT, get_gradZTT, get_ZTT_mineig_grad

feasiblept = None

def Lags_normsqr(Lags):
    return np.sum(Lags*Lags), 2*Lags

def Lags_normsqr_Hess_np(Lags):
    return 2*np.eye(len(Lags))

def get_ZTT_mineig(n_S, incLags, include, O, gradZTT):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    eigw = la.eigvalsh(ZTT)
    if eigw[0]>=0:
        global feasiblept
        feasiblept = incLags
        raise ValueError('found a feasible point')
    return eigw[0]


def get_ZTT_gradmineig(n_S, incLags, include, O, gradZTT):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    mineigJac = np.zeros((1,len(incLags)))
    mineigJac[0,:] = get_ZTT_mineig_grad(ZTT, gradZTT)[include]
    return mineigJac


def spatialProjopt_find_feasiblept(n_S, Lagnum, include, O, gradZTT):
    incLagnum = np.sum(include)
    initincLags = np.random.rand(incLagnum)
    
    mineigincfunc = lambda incL: get_ZTT_mineig(n_S, incL, include, O, gradZTT)
    Jacmineigincfunc = lambda incL: get_ZTT_gradmineig(n_S, incL, include, O, gradZTT)
    
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

