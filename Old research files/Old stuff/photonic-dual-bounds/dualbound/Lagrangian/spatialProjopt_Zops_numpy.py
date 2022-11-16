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


def Z_TT(Lags, O, UPlist):
    #P0 is identity and UP0 is the original U matrix
    ZTT = np.zeros_like(O, dtype=np.complex)
    ZTT[:,:] = O[:,:]
    for i in range(len(UPlist)):
        SymUP = (UPlist[i]+UPlist[i].conj().T)/2
        AsymUP = (UPlist[i]-UPlist[i].conj().T)/(2j)
        ZTT += Lags[2*i]*SymUP + Lags[2*i+1]*AsymUP
    return ZTT

def grad_Z_TT(Lags, UPlist):
    gradZ = []
    for i in range(len(UPlist)):
        SymUP = (UPlist[i]+UPlist[i].conj().T)/2
        AsymUP = (UPlist[i]-UPlist[i].conj().T)/(2j)
        gradZ.append(SymUP)
        gradZ.append(AsymUP)
    return gradZ


def check_spatialProj_Lags_validity(Lags, Olist, UPlistlist):
    modenum = len(Olist)
    mineig = np.inf
    for mode in range(modenum):
        ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
        eigZTT = np.linalg.eigvalsh(ZTT)
        
        if eigZTT[0]<0:
            print('mineig', eigZTT[0])
            return eigZTT[0]
        mineig = min(mineig,eigZTT[0])
    return mineig


def find_singular_ZTT_eigv(Lags, Olist, UPlistlist):
    modenum = len(Olist)
    mineigw = np.inf
    mineigv = np.zeros(Olist[0].shape[0])
    
    modemineig = -1
    for i in range(modenum):
        ZTT = Z_TT(Lags, Olist[i], UPlistlist[i])
        eigw, eigv = np.linalg.eigh(ZTT)
        if eigw[0]<=0:
            modemineig = i
            mineigv = eigv[:,0]
            return modemineig, mineigv
        elif eigw[0]<mineigw:
            mineigw = eigw[0]
            mineigv = eigv[:,0]
            modemineig = i
    return modemineig, mineigv


def get_ZTT_mineig(Lags, Olist, UPlistlist, eigvals_only=False):
    modenum = len(Olist)
    mineigw = np.inf
    modemineig = -1
    
    if eigvals_only:
        for mode in range(modenum):
            ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
            eigw = np.linalg.eigvalsh(ZTT)
            if eigw[0]<=0:
                return mode, eigw[0]
            elif eigw[0]<mineigw:
                mineigw = eigw[0]
                modemineig = mode
        return modemineig, mineigw
    else:
        for mode in range(modenum):
            ZTT = Z_TT(Lags, Olist[mode], UPlistlist[mode])
            eigw, eigv = np.linalg.eigh(ZTT)
            if eigw[0]<=0:
                return mode, eigw[0], eigv[:,0]
            elif eigw[0]<mineigw:
                mineigw = eigw[0]
                mineigv = eigv[:,0]
                modemineig = mode
        return modemineig, mineigw, mineigv

def get_inc_ZTT_mineig(incLags, include, Olist, UPlistlist, eigvals_only=False):
    Lags = np.zeros(len(include))
    Lags[include] = incLags[:]
    return get_ZTT_mineig(Lags, Olist, UPlistlist, eigvals_only=eigvals_only)


###method for finding derivatives of mineig of ZTT, to use for phase I (entering domain of duality) of optimization

def get_ZTT_mineig_grad(ZTT, gradZTT):
    eigw, eigv = np.linalg.eigh(ZTT)
    eiggrad = np.zeros(len(gradZTT))
    
    for i in range(len(eiggrad)):
        eiggrad[i] = np.real(np.vdot(eigv[:,0], gradZTT[i] @ eigv[:,0]))
    return eiggrad
