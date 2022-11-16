#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 31 12:24:53 2020

@author: pengning
This is part of the grad/Hess engine for spatial projection constraints + N sources 
Constraints take the form <S_j1|P_i|T_j2>-<T_j1|UP_i|T_j2> where j1,j2 \in {1,..,N}, 
and P_i spans the different spatial projections
The Lagrangian multipliers are distributed in the order 
alphaP0_Re11, alphaP0_Re12, ..., alphaP0_Re1N, alphaP0_Re21, ..., alphaP0_Re2N, ..., .., alphaP0_ReNN,
alphaP0_Im11, alphaP0_Im12, ..., alphaP0_Im1N, alphaP0_Im21, ..., alphaP0_Im2N, ..., .., alphaP0_ImNN,
alphaP1_Re11, ... and so on. Each spatial region gives 2*N^2 multipliers
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def get_gradZTT(n_S, UPlist):
    n_basis = UPlist[0].shape[0]
    shape_ZTT = (n_S*n_basis, n_S*n_basis)
    n_cplx_projLags = n_S**2
    gradZ = [0] * (2 * len(UPlist) * n_S**2)

    for l in range(len(UPlist)):
        UP = UPlist[l]
        UPH = UP.T.conj()
        SymUP = (UP+UPH)/2
        AsymUP = (UP-UPH)/(2j)
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j
                gradZ[ind_re] = np.zeros(shape_ZTT, dtype=np.complex)
                gradZ[ind_im] = np.zeros(shape_ZTT, dtype=np.complex)
                if i==j:
                    gradZ[ind_re][i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = SymUP
                    gradZ[ind_im][i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] = AsymUP
                else:
                    gradZ[ind_re][i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = UP/2
                    gradZ[ind_re][j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = UPH/2
                    gradZ[ind_im][i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] = -1j*UP/2
                    gradZ[ind_im][j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] = 1j*UPH/2
    
    return gradZ

def get_ZTT(n_S, Lags, O, gradZTT):
    ZTT = O.copy()
    for i in range(len(Lags)):
        #print('ZTT shape', ZTT.shape)
        #print('gradZTT[i].shape', gradZTT[i].shape, flush=True)
        ZTT += Lags[i] * gradZTT[i]
    return ZTT


def get_gradZTS_S(n_S, Si_st, Pdeslist):
    gradZTS_S = []

    num_basisS, num_basisT = Pdeslist[0].shape #account for possibility of non-square Pdes due to e.g. dimension reduction from div free constraint
    shape_ZTS = (n_S*num_basisT,n_S*num_basisS)
    gradZTS_S = [0] * (2 * len(Pdeslist) * n_S**2)

    n_cplx_projLags = n_S**2
    
    for l in range(len(Pdeslist)):
        ind_offset_Lags = l * 2 * n_cplx_projLags
        PH = Pdeslist[l].T.conj()
        for i in range(n_S):
            for j in range(n_S):
                ind_re = ind_offset_Lags + i*n_S + j
                ind_im = ind_offset_Lags + n_cplx_projLags + i*n_S + j

                gradZTS_re = np.zeros(shape_ZTS, dtype=np.complex)
                gradZTS_re[j*num_basisT:(j+1)*num_basisT,i*num_basisS:(i+1)*num_basisS] = PH / 2

                gradZTS_S_re = gradZTS_re @ Si_st
                gradZTS_S_im = 1j * gradZTS_S_re

                gradZTS_S[ind_re] = gradZTS_S_re
                gradZTS_S[ind_im] = gradZTS_S_im

    return gradZTS_S


def check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT):
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1


def check_spatialProj_incLags_validity(n_S, incLags, include, O, gradZTT):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(n_S, Lags, O, gradZTT)


def get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=False):
    ZTT = get_ZTT(n_S, Lags, O, gradZTT)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]


def get_inc_ZTT_mineig(n_S, incLags, include, O, gradZTT, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(n_S, Lags, O, gradZTT, eigvals_only=eigvals_only)

###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = la.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad
