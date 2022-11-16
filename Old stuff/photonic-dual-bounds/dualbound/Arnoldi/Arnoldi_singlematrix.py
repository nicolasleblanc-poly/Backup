#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 19:15:58 2020

@author: pengning

Try doing Arnoldi on dense Green's function matrix to extract efficient representation
for the purpose of bounds calculations
"""

import numpy as np
import scipy.linalg as la


def Green_Arnoldi_1S_oneshot(Gmat, S, vecnum):
    """
    does Arnoldi procedure on Gmat with single initial vector S
    generates a total of vecnum unit vectors
    outputs the resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    unitS = S / la.norm(S)
    unitvecs = np.zeros((len(S),vecnum+1), dtype=np.complex)
    unitvecs[:,0] = unitS #initial basis vector just the source
    unitvecs[:,1] = Gmat @ unitS #for iteration efficiency, last element of unitvecs is Gmat @ unitvecs[-2]
    A_Gmat = np.zeros((vecnum,vecnum), dtype=np.complex)
    A_Gmat[0,0] = np.vdot(unitS,unitvecs[:,1])
    for i in range(1,vecnum):
        img = unitvecs[:,i]
        for j in range(i):
            img -= A_Gmat[j,i-1] * unitvecs[:,j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[:,i] = img
        
        unitvecs[:,i+1] = Gmat @ img
        for j in range(i+1):
            A_Gmat[j,i] = np.vdot(unitvecs[:,j],unitvecs[:,i+1])
            A_Gmat[i,j] = np.vdot(unitvecs[:,i],Gmat @ unitvecs[:,j])
    
    return A_Gmat, unitvecs[:,:-1] #the last column of unitvecs is not orthogonalized and normalized and is left out


def Green_Arnoldi_multiS_oneshot(Gmat, multiS, vecnum):
    """
    does Arnoldi procedure on Gmat with multiple initial vectors stored in columns of multiS
    generates a total of vecnum unit vectors
    outputs the resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    try:
        n_S = multiS.shape[1]
    except IndexError:
        n_S = 1
        multiS = np.reshape(multiS, (len(multiS),1))
    if vecnum<n_S:
        raise ValueError('vecnum is smaller than number of initial vectors')
    Q,_ = la.qr(multiS, mode='economic')
    unitvecs = np.zeros((multiS.shape[0],vecnum+n_S), dtype=np.complex)
    unitvecs[:,:n_S] = Q
    unitvecs[:,n_S:2*n_S] = Gmat @ Q
    A_Gmat = np.zeros((vecnum,vecnum), dtype=np.complex)
    A_Gmat[:n_S,:n_S] = Q.conj().T @ unitvecs[:,n_S:2*n_S]
    
    for i in range(n_S,vecnum):
        img = unitvecs[:,i]
        for j in range(i):
            img -= A_Gmat[j,i-n_S] * unitvecs[:,j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[:,i] = img
        
        unitvecs[:,i+n_S] = Gmat @ img
        for j in range(i+1):
            A_Gmat[j,i] = np.vdot(unitvecs[:,j],unitvecs[:,i+n_S])
            A_Gmat[i,j] = np.vdot(unitvecs[:,i],Gmat @ unitvecs[:,j])
    
    return A_Gmat, unitvecs[:,:vecnum]



def Green_Arnoldi_1S_Uconverge(invchi, Gmat, S, Unormtol=1e-6, veclim=3, delveclim=2):
    """
    does Arnoldi procedure on Gmat with single initial vector S
    termination of Arnoldi when the image of S under Uconj = (1/chi)-Gmat converges to relative Unormtol
    outputs resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    unitS = S / la.norm(S)
    unitvecs = [unitS]
    unitvecs.append(Gmat @ unitS)
    A_Gmat = np.zeros_like(Gmat, dtype=np.complex)
    A_Gmat[0,0] = np.vdot(unitS, unitvecs[1])
    Uconj = invchi*np.eye(1) - A_Gmat[:1,:1]
    prevUnorm = 1.0/Uconj[0,0]
    
    i=1
    while i<veclim and i<Gmat.shape[0]:
        img = unitvecs[i]
        for j in range(i):
            img -= A_Gmat[j,i-1] * unitvecs[j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[i] = img
        
        unitvecs.append(Gmat @ img)
        for j in range(i+1):
            A_Gmat[j,i] = np.vdot(unitvecs[j],unitvecs[i+1])
            A_Gmat[i,j] = np.vdot(unitvecs[i],Gmat @ unitvecs[j])
        
        i += 1
        if i==veclim: #check convergence
            Uconj = invchi*np.eye(i) - A_Gmat[:i,:i]
            b = np.zeros((i,1))
            b[0] = 1.0
            x = la.solve(Uconj,b)
            Unorm = la.norm(x)
            if np.abs(Unorm-prevUnorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return A_Gmat[:i,:i], np.stack(unitvecs[:i], axis=1)


def Green_Arnoldi_multiS_Uconverge(invchi, Gmat, multiS, Unormtol=1e-6, veclim=None, delveclim=None):
    """
    does Arnoldi procedure on Gmat with multiple initial vectors stored in columns of multiS
    termination of Arnoldi when the image of multiS under Uconj = (1/chi)-Gmat converges to relative Unormtol
    outputs resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    try:
        n_S = multiS.shape[1]
    except IndexError:
        n_S = 1
        multiS = np.reshape(multiS, (len(multiS),1))

    if Gmat.shape[0]<n_S:
        raise ValueError('number of initial vectors larger than dimension of vector space')
        
    if veclim is None:
        veclim = 2*n_S
    if delveclim is None:
        delveclim = n_S
        
    Q,_ = la.qr(multiS, mode='economic')
    unitvecs = np.zeros((multiS.shape[0],2*n_S), dtype=np.complex)
    unitvecs[:,:n_S] = Q
    unitvecs[:,n_S:2*n_S] = Gmat @ Q
    A_Gmat = np.zeros_like(Gmat, dtype=np.complex)
    A_Gmat[:n_S,:n_S] = Q.conj().T @ unitvecs[:,n_S:2*n_S]
    
    unitvecs = list(unitvecs.T) #convert unitvecs from matrix to list of np arrays for ease of iteration
    Uconj = invchi*np.eye(n_S) - A_Gmat[:n_S,:n_S]
    b0 = np.eye(n_S)
    x0 = la.solve(Uconj,b0)
    prevUnorm = la.norm(x0, axis=0)
    
    i = n_S
    while i<veclim and i<Gmat.shape[0]:
        img = unitvecs[i]
        for j in range(i):
            img -= A_Gmat[j,i-n_S] * unitvecs[j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[i] = img
        
        unitvecs.append(Gmat @ img)
        for j in range(i+1):
            A_Gmat[j,i] = np.vdot(unitvecs[j],unitvecs[i+n_S])
            A_Gmat[i,j] = np.vdot(unitvecs[i],Gmat @ unitvecs[j])
        
        i += 1
        if i==veclim: #check convergence
            Uconj = invchi*np.eye(i) - A_Gmat[:i,:i]
            b = np.zeros((i,n_S))
            b[:n_S,:n_S] = b0
            x = la.solve(Uconj,b)
            Unorm = la.norm(x, axis=0)
            if max(np.abs(Unorm-prevUnorm) - np.abs(Unorm)*Unormtol)>0: #if one of the images of the initial vectors under U has not converged, keep going
                veclim += delveclim
                prevUnorm = Unorm
    
    return A_Gmat[:i,:i], np.stack(unitvecs[:i], axis=1)


def UP_Green_Arnoldi_multiS_Uconverge(invchi, Proj, Gmat, multiS, Unormtol=1e-6, veclim=None, delveclim=None):
    """
    does Arnoldi procedure on Gmat with multiple initial vectors stored in columns of multiS
    termination of Arnoldi when the image of multiS under UP = ((1/chi)-Gmat).conj().T @ Proj converges to relative Unormtol
    outputs resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    try:
        n_S = multiS.shape[1]
    except IndexError:
        n_S = 1
        multiS = np.reshape(multiS, (len(multiS),1))

    if Gmat.shape[0]<n_S:
        raise ValueError('number of initial vectors larger than dimension of vector space')
        
    if veclim is None:
        veclim = 2*n_S
    if delveclim is None:
        delveclim = n_S
    
    UP = (invchi*np.eye(Gmat.shape[0]) - Gmat).conj().T @ Proj
    
    Q,_ = la.qr(multiS, mode='economic')
    unitvecs = np.zeros((multiS.shape[0],2*n_S), dtype=np.complex)
    unitvecs[:,:n_S] = Q
    unitvecs[:,n_S:2*n_S] = UP @ Q
    A_UP = np.zeros_like(UP, dtype=np.complex)
    A_UP[:n_S,:n_S] = Q.conj().T @ unitvecs[:,n_S:2*n_S]
    
    unitvecs = list(unitvecs.T) #convert unitvecs from matrix to list of np arrays for ease of iteration
    b0 = np.eye(n_S)
    x0 = la.solve(A_UP[:n_S,:n_S],b0)
    prevUPnorm = la.norm(x0, axis=0)
    
    i = n_S
    while i<veclim and i<UP.shape[0]:
        img = unitvecs[i]
        for j in range(i):
            img -= A_UP[j,i-n_S] * unitvecs[j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[i] = img
        
        unitvecs.append(UP @ img)
        for j in range(i+1):
            A_UP[j,i] = np.vdot(unitvecs[j],unitvecs[i+n_S])
            A_UP[i,j] = np.vdot(unitvecs[i],UP @ unitvecs[j])
        
        i += 1
        if i==veclim: #check convergence
            b = np.zeros((i,n_S))
            b[:n_S,:n_S] = b0
            x = la.solve(A_UP[:i,:i],b)
            UPnorm = la.norm(x, axis=0)
            if max(np.abs(UPnorm-prevUPnorm) - np.abs(UPnorm)*Unormtol)>0: #if one of the images of the initial vectors under U has not converged, keep going
                veclim += delveclim
                prevUPnorm = UPnorm
    
    unitvecs = np.stack(unitvecs[:i], axis=1)
    A_Gmat = unitvecs.conj().T @ Gmat @ unitvecs
    return A_UP[:i,:i], A_Gmat, unitvecs



def split_basis_get_Uinv_norm(P,U, unitvecs, multiS):
    """
    split the unitvectors stored in the columns of unitvecs into the projection region and its complement
    re-orthonormalize within each region to get new partial basis to expand U
    get the norm of U^{-1} @ multiS using this new partial basis for determining convergence of Arnoldi procedure
    """
    P_vec = P @ unitvecs
    PC_vec = unitvecs - P_vec
    
    QP, _ = la.qr(P_vec, mode='economic')
    QPC, _ = la.qr(PC_vec, mode='economic')
    
    A_unitvecs = np.zeros((P.shape[0], 2*unitvecs.shape[1]), dtype=np.complex)
    A_unitvecs[:,0::2] = QP[:,:]
    A_unitvecs[:,1::2] = QPC[:,:]
    
    A_U = A_unitvecs.conj().T @ U @ A_unitvecs

    P_multiS = P @ multiS
    PC_multiS = multiS - P_multiS
    split_multiS = np.zeros((multiS.shape[0], 2*multiS.shape[1]), dtype=np.complex)
    split_multiS[:,0::2] = P_multiS[:,:]
    split_multiS[:,1::2] = PC_multiS[:,:]
    A_multiS = A_unitvecs.conj().T @ split_multiS
    
    x = la.solve(A_U, A_multiS)
    Unorm = la.norm(x, axis=0)
    return A_unitvecs, Unorm

def Green_Arnoldi_Proj_multiS_Uconverge(invchi, Gmat, Proj, multiS, Unormtol=1e-6, veclim=None, delveclim=None):
    """
    does Arnoldi procedure on Gmat with multiple initial vectors stored in columns of multiS
    termination of Arnoldi when the image of split(multiS) under Uconj = (1/chi)-Gmat converges to relative Unormtol
    split(multiS) is set of vectors consisting of P @ multiS and (I-P) @ multiS
    outputs resulting Arnoldi approximation A_Gmat and matrix of Arnoldi vectors
    """
    try:
        n_S = multiS.shape[1]
    except IndexError:
        n_S = 1
        multiS = np.reshape(multiS, (len(multiS),1))

    if Gmat.shape[0]<n_S:
        raise ValueError('number of initial vectors larger than dimension of vector space')
        
    if veclim is None:
        veclim = 2*n_S
    if delveclim is None:
        delveclim = n_S
        
    Q,_ = la.qr(multiS, mode='economic')
    unitvecs = np.zeros((multiS.shape[0],2*n_S), dtype=np.complex)
    unitvecs[:,:n_S] = Q
    unitvecs[:,n_S:2*n_S] = Gmat @ Q
    A_Gmat = np.zeros_like(Gmat, dtype=np.complex)
    A_Gmat[:n_S,:n_S] = Q.conj().T @ unitvecs[:,n_S:2*n_S]
    
    Uconj = invchi*np.eye(Gmat.shape[0]) - Gmat
    A_unitvecs, prevUnorm = split_basis_get_Uinv_norm(Proj, Uconj, unitvecs[:,:n_S], multiS)
    unitvecs = list(unitvecs.T) #convert unitvecs from matrix to list of np arrays for ease of iteration

    i = n_S
    while i<veclim and i<Gmat.shape[0]:
        img = unitvecs[i]
        for j in range(i):
            img -= A_Gmat[j,i-n_S] * unitvecs[j] #orthogonalization
        img /= la.norm(img) #normalization
        unitvecs[i] = img
        
        unitvecs.append(Gmat @ img)
        for j in range(i+1):
            A_Gmat[j,i] = np.vdot(unitvecs[j],unitvecs[i+n_S])
            A_Gmat[i,j] = np.vdot(unitvecs[i],Gmat @ unitvecs[j])
        
        i += 1
        if i==veclim: #check convergence
            A_unitvecs, Unorm = split_basis_get_Uinv_norm(Proj, Uconj, np.stack(unitvecs[:i], axis=1), multiS)
            if max(np.abs(Unorm-prevUnorm) - np.abs(Unorm)*Unormtol)>0: #if one of the images of the initial vectors under U has not converged, keep going
                veclim += delveclim
                prevUnorm = Unorm
    
    #in the end use the splitted unitvecs as the final basis
    if i!=veclim:
        A_unitvecs, _ = split_basis_get_Uinv_norm(Proj, Uconj, np.stack(unitvecs[:i], axis=1), multiS)
    A_Gmat = A_unitvecs.conj().T @ Gmat @ A_unitvecs
    return A_Gmat, A_unitvecs
