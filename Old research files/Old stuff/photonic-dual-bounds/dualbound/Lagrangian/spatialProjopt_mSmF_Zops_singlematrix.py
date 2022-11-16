#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on May 18 2021

@author: pengning
This is part of the grad/Hess engine for spatial projection constraints + N sources 
Constraints take the form <S_j1|P_i|T_j2>-<T_j1|UP_i|T_j2> where j1,j2 \in {1,..,N}, 
and P_i spans the different spatial projections
The Lagrangian multipliers are distributed in the order 
alphaP0_Re11, alphaP0_Re12, ..., alphaP0_Re1N, alphaP0_Re21, ..., alphaP0_Re2N, ..., .., alphaP0_ReNN,
alphaP0_Im11, alphaP0_Im12, ..., alphaP0_Im1N, alphaP0_Im21, ..., alphaP0_Im2N, ..., .., alphaP0_ImNN,
alphaP1_Re11, ... and so on. Each spatial region gives 2*N^2 multipliers

allows for multiple frequencies represented by different U matrices stored in Ulist
splits UPlist into separate Ulist and Plist
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

def Z_TT(n_S, Lags, O, Ulist, Plist):
    """
    generate Z_TT block
    nS is # of sources involved in the problem
    U and P have the same dimensions as the # of spatial basis vectors nbasis
    O is the constant part of ZTT; O and ZTT have dimensions nS * nbasis
    """
    
    n_basis = Plist[0].shape[0]
    ZTT = np.zeros_like(O, dtype=np.complex)
    ZTT[:,:] = O[:,:]
    
    n_cplx_projLags = n_S**2
    
    for l in range(len(Plist)):
        P = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            UP = Ulist[i] @ P
            UPH = UP.T.conj()
            SymUP = (UP+UPH)/2
            AsymUP = (UP-UPH)/(2j)
            for j in range(n_S): #go through all the cross-source constraints
                alpha_re = Lags[ind_offset_Lags + i*n_S + j]
                alpha_im = Lags[ind_offset_Lags + n_cplx_projLags + i*n_S + j]
                if i==j: #if the constraint is within a single source
                    ZTT[i*n_basis:(i+1)*n_basis , i*n_basis:(i+1)*n_basis] += alpha_re*SymUP + alpha_im*AsymUP
                else:
                    ZTT[i*n_basis:(i+1)*n_basis , j*n_basis:(j+1)*n_basis] += (alpha_re - 1j*alpha_im)*UP/2
                    ZTT[j*n_basis:(j+1)*n_basis , i*n_basis:(i+1)*n_basis] += (alpha_re + 1j*alpha_im)*UPH/2
                    
    return ZTT


def grad_Z_TT(n_S, Lags, Ulist, Plist):
    n_basis = Plist[0].shape[0]
    shape_ZTT = (n_S*n_basis, n_S*n_basis)
    n_cplx_projLags = n_S**2
    gradZ = [0] * len(Lags)

    for l in range(len(Plist)):
        P = Plist[l]
        ind_offset_Lags = l * 2 * n_cplx_projLags
        for i in range(n_S):
            UP = Ulist[i] @ P
            UPH = UP.T.conj()
            SymUP = (UP+UPH)/2
            AsymUP = (UP-UPH)/(2j)
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


def check_spatialProj_Lags_validity(n_S, Lags, O, Ulist, Plist):

    ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1


def check_spatialProj_incLags_validity(n_S, incLags, include, O, Ulist, Plist):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(n_S, Lags, O, Ulist, Plist)


def get_ZTT_mineig(n_S, Lags, O, Ulist, Plist, eigvals_only=False):
    ZTT = Z_TT(n_S, Lags, O, Ulist, Plist)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]


def get_inc_ZTT_mineig(n_S, incLags, include, O, Ulist, Plist, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(n_S, Lags, O, Ulist, Plist, eigvals_only=eigvals_only)

###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = la.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad
