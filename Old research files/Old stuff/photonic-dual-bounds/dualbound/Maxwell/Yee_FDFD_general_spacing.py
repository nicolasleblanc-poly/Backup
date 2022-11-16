#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 14:16:30 2021

@author: pengning
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from dipole_2D import TMdipole_2D_SI, TEydipole_2D_SI, TExdipole_2D_SI

def get_pml_x(omega, x_prime, x_dual, Npmlx, m=3, lnR=-20):
    Nx = len(x_prime)
    pml_x_Hz = np.ones(Nx, dtype=np.complex128)
    pml_x_Ey = np.ones(Nx, dtype=np.complex128)
    
    if Npmlx==0:
        return pml_x_Hz, pml_x_Ey
        
    w_pml = x_prime[Npmlx]-x_prime[0]
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    #define left PML
    pml_x_Hz[:Npmlx] = 1.0 / (1.0 + 1j * sigma_max * ((w_pml-x_prime[:Npmlx]) / w_pml)**m)
    pml_x_Ey[:Npmlx] = 1.0 / (1.0 + 1j * sigma_max * ((w_pml-x_dual[:Npmlx]) / w_pml)**m)
    
    #define right PML
    pml_x_Hz[-Npmlx:] = 1.0 / (1.0 + 1j * sigma_max * ((x_prime[-Npmlx:]-x_prime[-Npmlx])/w_pml)**m)
    pml_x_Ey[-Npmlx:] = 1.0 / (1.0 + 1j * sigma_max * ((x_dual[-Npmlx:]-x_prime[-Npmlx])/w_pml)**m)
    
    return pml_x_Hz, pml_x_Ey


def get_pml_y(omega, y_prime, y_dual, Npmly, m=3, lnR=-20):
    Ny = len(y_prime)
    pml_y_Hz = np.ones(Ny, dtype=np.complex128)
    pml_y_Ex = np.ones(Ny, dtype=np.complex128)
    
    if Npmly==0:
        return pml_y_Hz, pml_y_Ex

    w_pml = y_prime[Npmly]-y_prime[0]
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    #define bottom PML
    pml_y_Hz[:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((w_pml-y_prime[:Npmly])/w_pml)**m)
    pml_y_Ex[:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((w_pml-y_dual[:Npmly])/w_pml)**m)
    
    #define top PML
    pml_y_Hz[-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y_prime[-Npmly:]-y_prime[-Npmly])/w_pml)**m)
    pml_y_Ex[-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y_dual[-Npmly:]-y_prime[-Npmly])/w_pml)**m)
    
    return pml_y_Hz, pml_y_Ex


def build_TE_vac_A(omega, x_prime, x_dual, y_prime, y_dual, Npmlx, Npmly):
    """
    construct TE FDFD system matrix for vacuum
    the ordering of the indices goes (x,y,Hz), (x,y,Ex), (x,y,Ey), (x,y+1,Hz), (x,y+1,Ex) , ...
    gridpoint coordinates given by x_prime, x_dual, y_prime, y_dual
    grid can thus be non-uniform
    for now default periodic boundary conditions with no phase shift
    """
    Nx = len(x_prime)
    Ny = len(y_prime)
    pml_x_Hz, pml_x_Ey = get_pml_x(omega, x_prime, x_dual, Npmlx)
    pml_y_Hz, pml_y_Ex = get_pml_y(omega, y_prime, y_dual, Npmly)

    #construct system matrix A
    A_data = []
    A_i = []
    A_j = []  #prepare to construct A matrix in COO format
    
    for cx in range(Nx):
        for cy in range(Ny):
            xyind = cx*Ny + cy
            
            if cx<Nx-1:
                cxp1 = cx+1
            else:
                cxp1 = 0
            
            if cx>0:
                cxm1 = cx-1
            else:
                cxm1 = Nx-1
                
            if cy<Ny-1:
                cyp1 = cy+1
            else:
                cyp1 = 0
                
            if cy>0:
                cym1 = cy-1
            else:
                cym1 = Ny-1
                
            xp1yind = cxp1*Ny + cy
            xm1yind = cxm1*Ny + cy
            xyp1ind = cx*Ny + cyp1
            xym1ind = cx*Ny + cym1
            
            
            Hzind = 3*xyind
            #construct Hz row
            i = Hzind
            A_i.append(i); A_j.append(i); A_data.append(-1j*omega) #diagonal
            
            jEx0 = 3*xym1ind + 1
            jEx1 = i + 1
            if cy>0:
                delta = y_dual[cy] - y_dual[cy-1] #Ex is situated on dual grid
            else:
                delta = y_dual[-1] - y_dual[-2]
            A_i.append(i); A_j.append(jEx0); A_data.append(pml_y_Hz[cy]/delta)
            A_i.append(i); A_j.append(jEx1); A_data.append(-pml_y_Hz[cy]/delta) #Ex part of curl E term
            
            jEy0 = i + 2
            jEy1 = 3*xp1yind + 2
            if cx<Nx-1:
                delta = x_dual[cx+1] - x_dual[cx]
            else:
                delta = x_dual[1] - x_dual[0]
            A_i.append(i); A_j.append(jEy0); A_data.append(-pml_x_Hz[cx]/delta) #Ey part of curl E term
            A_i.append(i); A_j.append(jEy1); A_data.append(pml_x_Hz[cx]/delta)
            
            #construct Ex row
            i = i+1 #Ex comes after Hz
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            
            jHz0 = Hzind
            jHz1 = 3*xyp1ind
            if cy<Ny-1:
                delta = y_prime[cy+1] - y_prime[cy]
            else:
                delta = y_prime[1] - y_prime[0]
            A_i.append(i); A_j.append(jHz0); A_data.append(-pml_y_Ex[cy]/delta)
            A_i.append(i); A_j.append(jHz1); A_data.append(pml_y_Ex[cy]/delta) #Hz curl
            
            #constraint Ey row
            i = i+1 #Ey comes after Ex
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            
            jHz0 = 3*xm1yind
            jHz1 = Hzind
            if cx>0:
                delta = x_prime[cx] - x_prime[cx-1]
            else:
                delta = x_prime[-1] - x_prime[-2]
            A_i.append(i); A_j.append(jHz0); A_data.append(pml_x_Ey[cx]/delta)
            A_i.append(i); A_j.append(jHz1); A_data.append(-pml_x_Ey[cx]/delta) #Hz curl
            
    
    A = sp.coo_matrix((A_data, (A_i,A_j)), shape=(3*Nx*Ny,3*Nx*Ny))
    
    return A.tocsc()


def extract_TM_Maxwell_from_A(A):
    #take note of fact that field duality transformation is H=-E'  E=H'  M=J'  J=-M'
    #here we will make use that A_HE and A_EH are all representations of the curl operator
    #for TM, the Yee grid goes like (0,0,Ez) (0,0,Hx) (0,0,Hy), (0,1,Ez), ...
    omega = np.abs(A[0,0])
    inds = np.arange(A.shape[0])
    E_mask = np.zeros(A.shape[0], dtype=np.bool)
    E_mask[::3] = True
    E_inds = inds[E_mask]
    H_inds = inds[np.logical_not(E_mask)]
    
    A_HE = A[H_inds,:][:,E_inds]
    A_EH = A[E_inds,:][:,H_inds]
    
    M = A_EH @ A_HE - omega**2*sp.eye(A.shape[0]//3, format='csc')
    return M
    

def extract_TE_Maxwell_from_A(A, ordering='point'):
    omega = np.abs(A[0,0])
    inds = np.arange(A.shape[0])
    H_mask = np.zeros(A.shape[0], dtype=np.bool)
    H_mask[::3] = True
    H_inds = inds[H_mask]
    E_inds = inds[np.logical_not(H_mask)]
    
    A_HE = A[H_inds,:][:,E_inds]
    A_EH = A[E_inds,:][:,H_inds]
    
    M = A_EH @ A_HE - omega**2*sp.eye(2*A.shape[0]//3, format='csc')
    
    if ordering=='pol': #switch from (x,y,Ex), (x,y,Ey), ... ordering to (x,y,Ex), (x,y+1,Ex)... (x,y,Ey) ordering
        inds = np.arange(M.shape[0])
        shuffle = np.zeros_like(inds)
        shuffle[:M.shape[0]//2] = inds[::2]
        shuffle[M.shape[0]//2:] = inds[1::2]
        M = M[shuffle,:][:,shuffle]
        
    return M


def get_Yee_TE_Gddinv(wvlgth, x_prime, x_dual, y_prime, y_dual, Nx_pml, Ny_pml, designMask, ordering='point'):
    
    omega = 2*np.pi / wvlgth
    A = build_TE_vac_A(omega, x_prime, x_dual, y_prime, y_dual, Nx_pml, Ny_pml)
    M = extract_TE_Maxwell_from_A(A, ordering=ordering)
    
    Nx = len(x_prime); Ny = len(y_prime)
    full_designMask = np.zeros((2, Nx*Ny), dtype=np.bool)
    full_designMask[0,:] = designMask.flatten()
    full_designMask[1,:] = designMask.flatten()
    if ordering=='point':
        designInd = np.nonzero(full_designMask.T.flatten())[0]
        backgroundInd = np.nonzero(np.logical_not(full_designMask.T.flatten()))[0]
    else:
        designInd = np.nonzero(full_designMask.flatten())[0]
        backgroundInd = np.nonzero(np.logical_not(full_designMask.flatten()))[0]
    
    A = (M[:,backgroundInd])[backgroundInd,:]
    B = (M[:,designInd])[backgroundInd,:]
    C = (M[:,backgroundInd])[designInd,:]
    D = (M[designInd,:])[:,designInd]
    
    AinvB = spla.spsolve(A, B)
    MU_0 = 1.0
    Gfac = MU_0 / omega**2
    return (D - (C @ AinvB))*Gfac, M
    

def test_nonUniform_Yee_TE_dipole():
    """
    test dipole field calculation accuracy in comparison with analytic expressions
    use a non-uniform grid description with two grid-sizes
    """
    wvlgth = 1.0
    Lx_big = 1.0
    Lx_small = 2.0
    Lx_pml = 0.5
    Ly_big = 1.0
    Ly_small = 2.0
    Ly_pml = 0.5
    
    #adjust the discretization sizes to test
    dx_big = 0.02
    dx_small = 0.02
    
    dy_big = 0.02
    dy_small = 0.02
    
    Nx_pml = int(np.round(Lx_pml / dx_big))
    Nx_big = int(np.round(Lx_big / dx_big)) + Nx_pml
    Nx_small = int(np.round(Lx_small / dx_small))
    
    Ny_pml = int(np.round(Ly_pml / dy_big))
    Ny_big = int(np.round(Ly_big / dy_big)) + Ny_pml
    Ny_small = int(np.round(Ly_small / dy_small))
    
    Nx_tot = 2*Nx_big+Nx_small
    Ny_tot = 2*Ny_big+Ny_small
    
    print('dx_big', dx_big, 'dx_small', dx_small, 'dy_big', dy_big, 'dy_small', dy_small)
    print('Nx_big', Nx_big, 'Nx_small', Nx_small, 'Ny_big', Ny_big, 'Ny_small', Ny_small)
    
    #construct coordinates of non-uniform grid
    x_prime = np.zeros(Nx_tot, dtype=np.double)
    x_dual = np.zeros(Nx_tot, dtype=np.double)
    x_prime[0] = 0.0; x_dual[0] = -dx_big/2
    for i in range(1,Nx_tot):
        if i<Nx_big or i>=Nx_big+Nx_small:
            x_prime[i] = x_prime[i-1] + dx_big
            x_dual[i] = x_prime[i] - dx_big/2
        else:
            x_prime[i] = x_prime[i-1] + dx_small
            x_dual[i] = x_prime[i] - dx_small/2
            
    y_prime = np.zeros(Ny_tot, dtype=np.double)
    y_dual = np.zeros(Ny_tot, dtype=np.double)
    y_prime[0] = 0.0; y_dual[0] = dy_big/2
    for j in range(1,Ny_tot):
        if j<Ny_big or j>=Ny_big+Ny_small:
            y_prime[j] = y_prime[j-1] + dy_big
            y_dual[j] = y_prime[j] + dy_big/2
        else:
            y_prime[j] = y_prime[j-1] + dy_small
            y_dual[j] = y_prime[j] + dy_small/2
        
    omega = 2*np.pi / wvlgth
    A = build_TE_vac_A(omega, x_prime, x_dual, y_prime, y_dual, Nx_pml, Ny_pml)
    
    print('test TM source')
    b = np.zeros(A.shape[0], dtype=np.complex128)
    cx = Nx_tot // 2
    cy = Ny_tot // 2
    xyind = cx*Ny_tot + cy
    b[3*xyind] = 1.0 / (dx_small * dy_small) #adjust
    
    x = spla.spsolve(A, b)
    Hzfield = np.reshape(x[::3], (Nx_tot,Ny_tot))
    Exfield = np.reshape(x[1::3], (Nx_tot,Ny_tot))
    Eyfield = np.reshape(x[2::3], (Nx_tot,Ny_tot))
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Hzfield), cmap='RdBu')
    ax2.imshow(np.imag(Hzfield), cmap='RdBu')
    fig.suptitle('Hz field')
    plt.show()
    probe_Mx = [Nx_small//4, Nx_small//2, Nx_small//2 + Nx_big//2]
    probe_My = [Ny_small//4, Ny_small//2, Ny_small//2 + Ny_big//2]
    
    probe_Mx = [Nx_small//4, Nx_small//2, Nx_small//2 + Nx_big//2]
    probe_My = [Ny_small//4, Ny_small//2, Ny_small//2 + Ny_big//2]
    
    for i in range(len(probe_Mx)):
        Mx = probe_Mx[i]
        My = probe_My[i]
        Hz_fdfd = Hzfield[cx+Mx, cy+My]
        
        rho = np.hypot(x_prime[cx+Mx]-x_prime[cx], y_prime[cy+My]-y_prime[cy])
        Hz_analytic = TMdipole_2D_SI(wvlgth, rho, amp=-1.0) #H field source has a - sign
        print('Hz comparison', Hz_fdfd, Hz_analytic, 'rel diff', np.abs(Hz_fdfd-Hz_analytic)/np.abs(Hz_analytic))

    print('\n')
    print('test TE source')
    b = np.zeros(A.shape[0], dtype=np.complex128)
    cx = Nx_tot // 2
    cy = Ny_tot // 2
    xyind = cx*Ny_tot + cy
    b[3*xyind+1] = 1.0 / (dx_small * dy_small)
    
    x = spla.spsolve(A, b)
    Hzfield = np.reshape(x[::3], (Nx_tot,Ny_tot))
    Exfield = np.reshape(x[1::3], (Nx_tot,Ny_tot))
    Eyfield = np.reshape(x[2::3], (Nx_tot,Ny_tot))
    
    probe_Mx = [Nx_small//4, Nx_small//2, Nx_small//2 + Nx_big//2]
    probe_My = [Ny_small//4, Ny_small//2, Ny_small//2 + Ny_big//2]
    
    for i in range(len(probe_Mx)):
        Mx = probe_Mx[i]
        My = probe_My[i]
        Ex_fdfd = Exfield[cx+Mx, cy+My]
        Ey_fdfd = Eyfield[cx+Mx, cy+My]
        
        E_analytic = TExdipole_2D_SI(wvlgth, x_prime[cx+Mx]-x_prime[cx], y_dual[cy+My]-y_dual[cy])
        Ex_analytic = E_analytic[0]
        
        E_analytic = TExdipole_2D_SI(wvlgth, x_dual[cx+Mx]-x_prime[cx], y_prime[cy+My]-y_dual[cy])
        Ey_analytic = E_analytic[1]
        print('Ex comparison', Ex_fdfd, Ex_analytic, 'rel diff', np.abs(Ex_fdfd-Ex_analytic)/np.abs(Ex_analytic))
        print('Ey comparison', Ey_fdfd, Ey_analytic, 'rel diff', np.abs(Ey_fdfd-Ey_analytic)/np.abs(Ey_analytic))
        
     

def test_extraction_dipole():
    """
    test the extraction of the Maxwell curlcurl - omega^2/c^2 operator from the
    Yee A operator
    """
    wvlgth = 1.0
    Lx_big = 0.5
    Lx_small = 0.5
    Lx_pml = 0.5
    Ly_big = 0.5
    Ly_small = 0.5
    Ly_pml = 0.5
    
    dx_big = 0.05
    dx_small = 0.05
    
    dy_big = 0.05
    dy_small = 0.05
    
    Nx_pml = int(np.round(Lx_pml / dx_big))
    Nx_big = int(np.round(Lx_big / dx_big)) + Nx_pml
    Nx_small = int(np.round(Lx_small / dx_small))
    
    Ny_pml = int(np.round(Ly_pml / dy_big))
    Ny_big = int(np.round(Ly_big / dy_big)) + Ny_pml
    Ny_small = int(np.round(Ly_small / dy_small))
    
    Nx_tot = 2*Nx_big+Nx_small
    Ny_tot = 2*Ny_big+Ny_small
    
    print('dx_big', dx_big, 'dx_small', dx_small, 'dy_big', dy_big, 'dy_small', dy_small)
    print('Nx_big', Nx_big, 'Nx_small', Nx_small, 'Ny_big', Ny_big, 'Ny_small', Ny_small)
    
    #construct coordinates of non-uniform grid
    x_prime = np.zeros(Nx_tot, dtype=np.double)
    x_dual = np.zeros(Nx_tot, dtype=np.double)
    x_prime[0] = 0.0; x_dual[0] = -dx_big/2
    for i in range(1,Nx_tot):
        if i<Nx_big or i>=Nx_big+Nx_small:
            x_prime[i] = x_prime[i-1] + dx_big
            x_dual[i] = x_prime[i] - dx_big/2
        else:
            x_prime[i] = x_prime[i-1] + dx_small
            x_dual[i] = x_prime[i] - dx_small/2
            
    y_prime = np.zeros(Ny_tot, dtype=np.double)
    y_dual = np.zeros(Ny_tot, dtype=np.double)
    y_prime[0] = 0.0; y_dual[0] = dy_big/2
    for j in range(1,Ny_tot):
        if j<Ny_big or j>=Ny_big+Ny_small:
            y_prime[j] = y_prime[j-1] + dy_big
            y_dual[j] = y_prime[j] + dy_big/2
        else:
            y_prime[j] = y_prime[j-1] + dy_small
            y_dual[j] = y_prime[j] + dy_small/2
        
    omega = 2*np.pi / wvlgth
    A = build_TE_vac_A(omega, x_prime, x_dual, y_prime, y_dual, Nx_pml, Ny_pml)
    
    print('test TM extraction')
    M = extract_TM_Maxwell_from_A(A)
    
    print('TM M size', M.shape, M.shape[0]*M.shape[1], 'number of non-zero entries in M', M.count_nonzero())
    b = np.zeros(M.shape[0], dtype=np.complex128)
    cx = Nx_tot // 2
    cy = Ny_tot // 2
    xyind = cx*Ny_tot + cy
    b[xyind] = 1j*omega / (dx_small * dy_small)
    
    x = spla.spsolve(M, b)

    Ezfield = np.reshape(x, (Nx_tot,Ny_tot))
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Ezfield), cmap='RdBu')
    ax2.imshow(np.imag(Ezfield), cmap='RdBu')
    fig.suptitle('Ez field')
    plt.show()
    probe_Mx = [Nx_small//4, Nx_small//2, Nx_small//2 + Nx_big//2]
    probe_My = [Ny_small//4, Ny_small//2, Ny_small//2 + Ny_big//2]
    
    for i in range(len(probe_Mx)):
        Mx = probe_Mx[i]
        My = probe_My[i]
        Ez_fdfd = Ezfield[cx+Mx, cy+My]
        
        rho = np.hypot(x_prime[cx+Mx]-x_prime[cx], y_prime[cy+My]-y_prime[cy])
        Ez_analytic = TMdipole_2D_SI(wvlgth, rho, amp=1.0) #H field source has a - sign
        print('Ez comparison', Ez_fdfd, Ez_analytic, 'rel diff', np.abs(Ez_fdfd-Ez_analytic)/np.abs(Ez_analytic))


    print('test TE extraction')
    #M = extract_TE_Maxwell_from_A(A, ordering='pol')
    designMask = np.zeros((Nx_tot, Ny_tot), dtype=np.bool)
    designMask[Nx_big:Nx_big+Nx_small, Ny_big:Ny_big+Ny_small] = True
    Gddinv, M = Gddinv, M = get_Yee_TE_Gddinv(wvlgth, x_prime, x_dual, y_prime, y_dual, Nx_pml, Ny_pml, designMask, ordering='pol')
    print('TE M size', M.shape, M.shape[0]*M.shape[1], 'number of non-zero entries in M', M.count_nonzero())
    b = np.zeros(M.shape[0], dtype=np.complex128)
    cx = Nx_tot // 2
    cy = Ny_tot // 2
    xyind = cx*Ny_tot + cy
    b[xyind] = 1j*omega / (dx_small * dy_small)
    
    x = spla.spsolve(M, b)
    Exfield = np.reshape(x[:len(x)//2], (Nx_tot,Ny_tot))
    Eyfield = np.reshape(x[len(x)//2:], (Nx_tot,Ny_tot))
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Exfield), cmap='RdBu')
    ax2.imshow(np.imag(Exfield), cmap='RdBu')
    fig.suptitle('Ex field')
    plt.show()
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Eyfield), cmap='RdBu')
    ax2.imshow(np.imag(Eyfield), cmap='RdBu')
    fig.suptitle('Ey field')
    plt.show()
    
    probe_Mx = [Nx_small//4, Nx_small//2, Nx_small//2 + Nx_big//2]
    probe_My = [Ny_small//4, Ny_small//2, Ny_small//2 + Ny_big//2]
    
    for i in range(len(probe_Mx)):
        Mx = probe_Mx[i]
        My = probe_My[i]
        Ex_fdfd = Exfield[cx+Mx, cy+My]
        Ey_fdfd = Eyfield[cx+Mx, cy+My]
        
        E_analytic = TExdipole_2D_SI(wvlgth, x_prime[cx+Mx]-x_prime[cx], y_dual[cy+My]-y_dual[cy])
        Ex_analytic = E_analytic[0]
        
        E_analytic = TExdipole_2D_SI(wvlgth, x_dual[cx+Mx]-x_prime[cx], y_prime[cy+My]-y_dual[cy])
        Ey_analytic = E_analytic[1]
        print('Ex comparison', Ex_fdfd, Ex_analytic, 'rel diff', np.abs(Ex_fdfd-Ex_analytic)/np.abs(Ex_analytic))
        print('Ey comparison', Ey_fdfd, Ey_analytic, 'rel diff', np.abs(Ey_fdfd-Ey_analytic)/np.abs(Ey_analytic))
    
    testE = np.zeros(2*Nx_small*Ny_small, dtype=np.complex)
    testE[:Nx_small*Ny_small] = Exfield[designMask]
    testE[Nx_small*Ny_small:] = Eyfield[designMask]
    
    testJ = -1j * (2*np.pi/wvlgth) * Gddinv @ testE
    testJx = np.reshape(testJ[:Nx_small*Ny_small], (Nx_small,Ny_small))
    testJy = np.reshape(testJ[Nx_small*Ny_small:], (Nx_small,Ny_small))
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(testJx), cmap='RdBu')
    ax2.imshow(np.imag(testJx), cmap='RdBu')
    plt.title('Jx')
    plt.show()
    print('max real Jx', np.max(np.abs(np.real(testJx))), 'max imag Jx', np.max(np.abs(np.imag(testJx))))
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(testJy), cmap='RdBu')
    ax2.imshow(np.imag(testJy), cmap='RdBu')
    plt.title('Jy')
    plt.show()
    print('max real Jy', np.max(np.abs(np.real(testJy))), 'max imag Jy', np.max(np.abs(np.imag(testJy))))
    