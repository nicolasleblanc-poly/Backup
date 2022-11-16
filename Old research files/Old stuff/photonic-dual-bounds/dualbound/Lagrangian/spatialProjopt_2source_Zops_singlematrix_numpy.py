#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This is part of the grad/Hess engine for spatial projection constraints + 2 sources
for a general basis where the relevant operators are all represented by a single matrix
Constraints take the form <S_j1|P_i|T_j2>-<T_j1|UP_i|T_j2> where j1,j2 \in {1,2}, 
and P_i spans the different spatial projections
The Lagrangian multipliers are distributed in the order 
alphaP0_Re11, alphaP0_Re12, alphaP0_Re21, alphaP0_Re22, alphaP0_Im11, alphaP0_Im12, alphaP0_Im21, alphaP0_Im22,
alphaP1_Re11, ... and so on 
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def Z_TT(Lags, O, UPlist):
    """
    generate ZTT block for a given mode family
    O is the constant part of ZTT and has the same dimensions as ZTT
    U and P have the same dimensions as the # of spatial basis vectors
    """
    num_basis = UPlist[0].shape[0]
    ZTT = np.zeros_like(O, dtype=np.complex)
    ZTT[:,:] = O[:,:]
    #print('in getZTT, Omat')
    #plt.figure()
    #plt.imshow(np.abs(O))
    #plt.show()
    for i in range(len(UPlist)):
        UP = UPlist[i]
        UPH = UP.T.conj()
        SymUP = (UP+UPH)/2
        AsymUP = (UP-UPH)/(2j)
        alpha_re11 = Lags[8*i]; alpha_im11 = Lags[8*i+4]
        alpha_re12 = Lags[8*i+1]; alpha_im12 = Lags[8*i+5]
        alpha_re21 = Lags[8*i+2]; alpha_im21 = Lags[8*i+6]
        alpha_re22 = Lags[8*i+3]; alpha_im22 = Lags[8*i+7]
        #11 contribution
        ZTT[:num_basis,:num_basis] += alpha_re11*SymUP + alpha_im11*AsymUP
        #22 contribution
        ZTT[num_basis:,num_basis:] += alpha_re22*SymUP + alpha_im22*AsymUP
        #12 contribution
        ZTT[:num_basis,num_basis:] += (alpha_re12 - 1j*alpha_im12)*UP/2
        ZTT[num_basis:,:num_basis] += (alpha_re12 + 1j*alpha_im12)*UPH/2
        #21 contribution
        ZTT[:num_basis,num_basis:] += (alpha_re21 + 1j*alpha_im21)*UPH/2
        ZTT[num_basis:,:num_basis] += (alpha_re21 - 1j*alpha_im21)*UP/2
    
    #print('colormap of ZTT')
    #plt.figure()
    #plt.imshow(np.abs(ZTT))
    #plt.show()
    return ZTT


def grad_Z_TT(Lags, UPlist):
    num_basis = UPlist[0].shape[0]
    shape_ZTT = (2*num_basis, 2*num_basis)
    gradZ = []
    for i in range(len(UPlist)):
        UP = UPlist[i]
        UPH = UP.T.conj()
        SymUP = (UP+UPH)/2
        AsymUP = (UP-UPH)/(2j)
        gradReZTT11 = np.zeros(shape_ZTT, dtype=np.complex)
        gradImZTT11 = np.zeros(shape_ZTT, dtype=np.complex)
        gradReZTT11[:num_basis,:num_basis] = SymUP
        gradImZTT11[:num_basis,:num_basis] = AsymUP

        gradReZTT22 = np.zeros(shape_ZTT, dtype=np.complex)
        gradImZTT22 = np.zeros(shape_ZTT, dtype=np.complex)
        gradReZTT22[num_basis:,num_basis:] = SymUP
        gradImZTT22[num_basis:,num_basis:] = AsymUP

        gradReZTT12 = np.zeros(shape_ZTT, dtype=np.complex)
        gradImZTT12 = np.zeros(shape_ZTT, dtype=np.complex)
        gradReZTT12[:num_basis,num_basis:] = UP/2
        gradReZTT12[num_basis:,:num_basis] = UPH/2
        gradImZTT12[:num_basis,num_basis:] = -1j*UP/2
        gradImZTT12[num_basis:,:num_basis] = 1j*UPH/2
        
        gradReZTT21 = np.zeros(shape_ZTT, dtype=np.complex)
        gradImZTT21 = np.zeros(shape_ZTT, dtype=np.complex)
        gradReZTT21[:num_basis,num_basis:] = UPH/2
        gradReZTT21[num_basis:,:num_basis] = UP/2
        gradImZTT21[:num_basis,num_basis:] = 1j*UPH/2
        gradImZTT21[num_basis:,:num_basis] = -1j*UP/2

        gradZ.append(gradReZTT11)
        gradZ.append(gradReZTT12)
        gradZ.append(gradReZTT21)
        gradZ.append(gradReZTT22)
        
        gradZ.append(gradImZTT11)
        gradZ.append(gradImZTT12)
        gradZ.append(gradImZTT21)
        gradZ.append(gradImZTT22)
    return gradZ


def check_spatialProj_Lags_validity(Lags, O, UPlist):

    ZTT = Z_TT(Lags, O, UPlist)
    try:
        _ = la.cholesky(ZTT)
        return 1
    except la.LinAlgError:
        return -1


def check_spatialProj_incLags_validity(incLags, include, O, UPlist):
    Lags = np.zeros(len(include), dtype=np.double)
    Lags[include] = incLags[:]
    return check_spatialProj_Lags_validity(Lags, O, UPlist)


def get_ZTT_mineig(Lags, O, UPlist, eigvals_only=False):
    ZTT = Z_TT(Lags, O, UPlist)
    if eigvals_only:
        eigw = la.eigvalsh(ZTT)
        return eigw[0]
    else:
        eigw, eigv = la.eigh(ZTT)
        return eigw[0], eigv[:,0]


def get_inc_ZTT_mineig(incLags, include, O, UPlist, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(Lags, O, UPlist, eigvals_only=eigvals_only)

###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = la.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad
