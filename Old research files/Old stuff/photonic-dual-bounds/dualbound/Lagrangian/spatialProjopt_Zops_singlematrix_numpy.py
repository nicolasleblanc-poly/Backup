#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 21:39:12 2020

@author: pengning

This is part of the grad/Hess engine for spatial projection versions of the 
original global constraint <S|T>-<T|U|T>. The Lagrangian multipliers are distributed in 
the order alphaP0_1, alphaP0_2, alphaP1_1, alphaP1_2 ... where P0 is just the identity
"""

import numpy as np
import scipy.linalg as la


def get_ZTT(Lags, O, gradZTT):
    ZTT = O.copy()
    for i in range(len(Lags)):
        ZTT += Lags[i] * gradZTT[i]
    return ZTT


def get_gradZTT(UPlist):
    gradZ = []
    for i in range(len(UPlist)):
        SymUP = (UPlist[i]+UPlist[i].conj().T)/2
        AsymUP = (UPlist[i]-UPlist[i].conj().T)/(2j)
        gradZ.append(SymUP)
        gradZ.append(AsymUP)
    return gradZ


def get_ZTS_S(Lags, O_lin, gradZTS_S):
    ZTS_S = O_lin.copy()
    for i in range(len(Lags)):
        ZTS_S += Lags[i] * gradZTS_S[i]
    return ZTS_S


def get_gradZTS_S(Si, Plist):
    gradZTS_S = []
    for i in range(len(Plist)):
        PdagS = Plist[i].conj().T @ Si
        gradZTS_S.append(PdagS/2)
        gradZTS_S.append(1j*PdagS/2)
    return gradZTS_S


def check_spatialProj_Lags_validity(Lags, O, gradZTT):

    ZTT = get_ZTT(Lags, O, gradZTT)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1


def check_spatialProj_incLags_validity(incLags, include, O, gradZTT):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(Lags, O, gradZTT)


def get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(Lags, O, gradZTT)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]


def get_inc_ZTT_mineig(incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(Lags, O, gradZTT, eigvals_only=eigvals_only)


###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = np.linalg.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad
