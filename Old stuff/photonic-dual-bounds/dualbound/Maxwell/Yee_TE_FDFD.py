#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 18:45:15 2021

2D E field in plane FDFD solver modeled after EMopt from the Yablonovich Group

uses dimensionless units, epsilon0=mu0=c0=1

Yee grid: Hz located at integer grid points, Ex located 0.5dy above corresponding Hz, Ey located 0.5dy left to corresponding Hz

@author: pengning
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt


def get_pml_x(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20):
    
    pml_x_Hz = np.ones((Nx,Ny), dtype=np.complex128)
    pml_x_Ey = np.ones((Nx,Ny), dtype=np.complex128)
    
    if Npmlx==0:
        return pml_x_Hz, pml_x_Ey
    
    x = np.arange(Nx)
    y = np.arange(Ny)
    
    X,Y = np.meshgrid(x,y, indexing='ij')
    w_pml = Npmlx * dx
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    #define left PML
    x = X[:Npmlx, :]
    pml_x_Hz[:Npmlx,:] = 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-x) / Npmlx)**m)
    pml_x_Ey[:Npmlx,:] = 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-x+0.5) / Npmlx)**m)
    
    #define right PML
    x = X[-Npmlx:, :]
    pml_x_Hz[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx)/Npmlx)**m)
    pml_x_Ey[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx-0.5)/Npmlx)**m)
    
    return pml_x_Hz, pml_x_Ey


def get_pml_y(omega, Nx, Ny, Npmly, dy, m=3, lnR=-20):
    
    pml_y_Hz = np.ones((Nx,Ny), dtype=np.complex128)
    pml_y_Ex = np.ones((Nx,Ny), dtype=np.complex128)
    
    if Npmly==0:
        return pml_y_Hz, pml_y_Ex
    
    x = np.arange(Nx)
    y = np.arange(Ny)
    
    X,Y = np.meshgrid(x,y, indexing='ij')
    w_pml = Npmly * dy
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    
    #define bottom PML
    y = Y[:,:Npmly]
    pml_y_Hz[:,:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((Npmly-y)/Npmly)**m)
    pml_y_Ex[:,:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((Npmly-y-0.5)/Npmly)**m)
    
    #define top PML
    y = Y[:,-Npmly:]
    pml_y_Hz[:,-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y-Ny+1+Npmly)/Npmly)**m)
    pml_y_Ex[:,-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y-Ny+1+Npmly+0.5)/Npmly)**m)
    
    return pml_y_Hz, pml_y_Ex



def build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy):
    """
    construct TE FDFD system matrix for vacuum
    the ordering of the indices goes (x,y,Hz), (x,y,Ex), (x,y,Ey), (x,y+1,Hz), (x,y+1,Ex) , ...
    for now default periodic boundary conditions with no phase shift
    """
    pml_x_Hz, pml_x_Ey = get_pml_x(omega, Nx, Ny, Npmlx, dx)
    pml_y_Hz, pml_y_Ex = get_pml_y(omega, Nx, Ny, Npmly, dy)
    
    A_data = []
    A_i = []
    A_j = []  #prepare to construct A matrix in COO format
    
    for cx in range(Nx):
        for cy in range(Ny):
            xyind = cx*Ny + cy
            if cx<Nx-1:
                xp1yind = (cx+1)*Ny + cy
            else:
                xp1yind = cy
            
            if cx>0:
                xm1yind = (cx-1)*Ny + cy
            else:
                xm1yind = (Nx-1)*Ny + cy
            
            if cy<Ny-1:
                xyp1ind = cx*Ny + cy + 1
            else:
                xyp1ind = cx*Ny
                
            if cy>0:
                xym1ind = cx*Ny + cy - 1
            else:
                xym1ind = cx*Ny + Ny - 1
            
            Hzind = 3*xyind
            #construct Hz row
            i = Hzind
            A_i.append(i); A_j.append(i); A_data.append(-1j*omega) #diagonal
            
            jEx0 = 3*xym1ind + 1
            jEx1 = i + 1
            A_i.append(i); A_j.append(jEx0); A_data.append(pml_y_Hz[cx,cy]/dy)
            A_i.append(i); A_j.append(jEx1); A_data.append(-pml_y_Hz[cx,cy]/dy) #Ex part of curl E term
            
            jEy0 = i + 2
            jEy1 = 3*xp1yind + 2
            A_i.append(i); A_j.append(jEy0); A_data.append(-pml_x_Hz[cx,cy]/dx) #Ey part of curl E term
            A_i.append(i); A_j.append(jEy1); A_data.append(pml_x_Hz[cx,cy]/dx)
            
            #construct Ex row
            i = i+1 #Ex comes after Hz
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            
            jHz0 = Hzind
            jHz1 = 3*xyp1ind
            A_i.append(i); A_j.append(jHz0); A_data.append(-pml_y_Ex[cx,cy]/dy)
            A_i.append(i); A_j.append(jHz1); A_data.append(pml_y_Ex[cx,cy]/dy) #Hz curl
            
            #constraint Ey row
            i = i+1 #Ey comes after Ex
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            
            jHz0 = 3*xm1yind
            jHz1 = Hzind
            A_i.append(i); A_j.append(jHz0); A_data.append(pml_x_Ey[cx,cy]/dx)
            A_i.append(i); A_j.append(jHz1); A_data.append(-pml_x_Ey[cx,cy]/dx) #Hz curl
            
    
    A = sp.coo_matrix((A_data, (A_i,A_j)), shape=(3*Nx*Ny,3*Nx*Ny))
    
    return A.tocsc()


def get_diagA_from_chigrid(omega, chi_x, chi_y):
    Nx,Ny = chi_x.shape
    diagA = np.zeros(3*Nx*Ny, dtype=np.complex128)
    diagA[1::3] = 1j*omega*chi_x.flatten()
    diagA[2::3] = 1j*omega*chi_y.flatten()
    return sp.diags(diagA)


def get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx,cy, pol, amp=1.0, Qabs=np.inf, chigrid=None):
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    A = build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    if not (chigrid is None):
        A += get_diagA_from_chigrid(omega, chigrid, chigrid)
        
    b = np.zeros(A.shape[0], dtype=np.complex128)
    
    xyind = cx*Ny + cy
    b[3*xyind+pol] = amp
    x = spla.spsolve(A,b)
    Hzfield = np.reshape(x[::3], (Nx,Ny))
    Exfield = np.reshape(x[1::3], (Nx,Ny))
    Eyfield = np.reshape(x[2::3], (Nx,Ny))
    
    return Hzfield, Exfield, Eyfield


def plot_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx,cy, pol, amp=1.0, Qabs=np.inf, chigrid=None):
    Hzfield, Exfield, Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx,cy, pol, amp=amp, Qabs=Qabs, chigrid=chigrid)
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Hzfield), cmap='RdBu')
    ax2.imshow(np.imag(Hzfield), cmap='RdBu')
    fig.suptitle('Hz field')
    plt.show()
    
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


def get_Yee_TE_Gddinv(wvlgth, dx, dy, Nx,Ny, Npmlx, Npmly, designMask, Qabs=np.inf, ordering='point'):
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    k0 = omega / 1
    A_Yee = build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    
    full_designMask = np.zeros((3, Nx*Ny), dtype=np.bool)
    full_designMask[1,:] = designMask.flatten()
    full_designMask[2,:] = designMask.flatten()
    designInd = np.nonzero(full_designMask.T.flatten())[0] #transpose due to order in which pol. locations are labeled
    backgroundInd = np.nonzero(np.logical_not(full_designMask.T.flatten()))[0]
    
    A = (A_Yee[:,backgroundInd])[backgroundInd,:]
    B = (A_Yee[:,designInd])[backgroundInd,:]
    C = (A_Yee[:,backgroundInd])[designInd,:]
    D = (A_Yee[designInd,:])[:,designInd]
    
    AinvB = spla.spsolve(A, B)
    ETA_0 = 1.0
    Gfac = 1j*ETA_0 / k0 #note different Gfac compared with TM case due to directly extracting from A matrix
    Gddinv = (D - (C @ AinvB))*Gfac
    if ordering=='pol':
        num_des = np.sum(designMask)
        inds = np.arange(2*num_des)
        shuffle = np.zeros_like(inds, dtype=np.int)
        shuffle[:num_des] = inds[::2]
        shuffle[num_des:] = inds[1::2]
        Gddinv = Gddinv[shuffle,:][:,shuffle]
    return Gddinv, A_Yee
    


def get_Yee_TE_GreenFcn(wvlgth, Gx,Gy, Npmlx, Npmly, dx,dy, Qabs=np.inf):
    """
    generate Green's function of a domain with shape (Gx,Gy)
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    gpwx = int(1.0/dx)
    gpwy = int(1.0/dy)
    Nx = 2*Gx-1 + gpwx//2 + 2*Npmlx
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, Qabs=Qabs)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, Qabs=Qabs)
    
    numCoord = Gx*Gy
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    
    for ix in range(Gx):
        for iy in range(Gy):
            x_Ex = x_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            x_Ey = x_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ex = y_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ey = y_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            
            xyind = ix*Gy + iy
            G[:numCoord,xyind] = x_Ex.flatten()
            G[numCoord:,xyind] = x_Ey.flatten()
            G[:numCoord,xyind+numCoord] = y_Ex.flatten()
            G[numCoord:,xyind+numCoord] = y_Ey.flatten()
    
    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G


def get_Yee_TE_masked_GreenFcn(wvlgth, Gx,Gy, Gmask, Npmlx, Npmly, dx,dy, Qabs=np.inf):
    """
    generate Green's function of a domain with shape specified by 2D boolean Gmask over a domain of size (Gx,Gy)
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    gpwx = int(1.0/dx)
    gpwy = int(1.0/dy)
    Nx = 2*Gx-1 + gpwx//2 + 2*Npmlx
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, Qabs=Qabs)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, Qabs=Qabs)
    
    numCoord = np.sum(Gmask)
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)

    G_idx = np.argwhere(Gmask)
    
    for i,idx in enumerate(G_idx):
        ulx = cx - idx[0]
        uly = cy - idx[1]
        
        x_Ex = x_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        x_Ey = x_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ex = y_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ey = y_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        
        G[:numCoord,i] = x_Ex[Gmask]
        G[numCoord:,i] = x_Ey[Gmask]
        G[:numCoord,i+numCoord] = y_Ex[Gmask]
        G[numCoord:,i+numCoord] = y_Ey[Gmask]
        
    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G


def get_Yee_periodicY_TE_masked_GreenFcn(wvlgth, Ny, Gx,Gy,Gmask, dx,dy, Npmlx, Qabs=np.inf, Npmlsep=None):
    """
    generate Green's function of a domain with shape specified by 2D boolean Gmask over a domain of shape (Gx,Gy), within a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    if Npmlsep is None:
        gpw = int(1.0/dx)
        Npmlsep = gpw//2
    Nx = 2*Gx-1 + 2*(Npmlx+Npmlsep)
    cx = Nx//2
    cy = Ny//2
    
    Npmly = 0 #periodic in Y direction
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2)
    
    x_Exfield = np.concatenate((x_Exfield,x_Exfield,x_Exfield), axis=1)
    x_Eyfield = np.concatenate((x_Eyfield,x_Eyfield,x_Eyfield), axis=1)
    y_Exfield = np.concatenate((y_Exfield,y_Exfield,y_Exfield), axis=1)
    y_Eyfield = np.concatenate((y_Eyfield,y_Eyfield,y_Eyfield), axis=1)
    cy += Ny #stack up 3 unitcells in Y direction for easy window sliding
    
    numCoord = np.sum(Gmask)
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    
    G_idx = np.argwhere(Gmask)
    
    for i,idx in enumerate(G_idx):
        ulx = cx - idx[0]
        uly = cy - idx[1]
        
        x_Ex = x_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        x_Ey = x_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ex = y_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ey = y_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        
        G[:numCoord,i] = x_Ex[Gmask]
        G[numCoord:,i] = x_Ey[Gmask]
        G[:numCoord,i+numCoord] = y_Ex[Gmask]
        G[numCoord:,i+numCoord] = y_Ey[Gmask]
    
    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G



def test_Gddinv(): #test Qabs
    wvlgth = 1.0
    Qabs = 1e4
    omega = (2*np.pi/wvlgth) * (1+1j/2/Qabs)
    k0 = omega / 1
    dL = 0.05
    Nx = 100
    Ny = 100
    Npml = 40
    Mx = 20
    My = 20
    Mx0 = (Nx-Mx)//2
    My0 = (Ny-My)//2
    designMask = np.zeros((Nx,Ny), dtype=np.bool)
    designMask[Mx0:Mx0+Mx,My0:My0+My] = True
    Gddinv, M_Yee = get_Yee_TE_Gddinv(wvlgth, dL, dL, Nx, Ny, Npml, Npml, designMask, Qabs=Qabs)
    
    #check Gdd definiteness
    Gdd = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs=Qabs)
    #Gddinv_pol, _ = get_Yee_TE_Gddinv(wvlgth, dL, dL, Nx, Ny, Npml, Npml, designMask, Qabs=Qabs, ordering='pol')
    #checkId = Gdd @ Gddinv_pol.todense()
    #plt.matshow(np.abs(checkId))
    AsymG = (Gdd-Gdd.conj().T) / 2j
    eigw = np.linalg.eigvalsh(AsymG)
    print('check AsymGdd definiteness', eigw[:5], eigw[-1])
    #add a phase projection
    Gdd_P = Gdd * (1-1j/2/Qabs)
    AsymG_P = (Gdd_P - Gdd_P.conj().T) / 2j
    eigw = np.linalg.eigvalsh(AsymG_P)
    print('check AsymG_P definiteness', eigw[:5], eigw[-1])
    
    
    Hzfield, Exfield, Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npml, Npml, dL, dL, Nx//2, Ny//2, 2, Qabs=Qabs)
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Exfield), cmap='RdBu')
    ax2.imshow(np.imag(Exfield), cmap='RdBu')
    plt.title('Ex')
    plt.show()
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Eyfield), cmap='RdBu')
    ax2.imshow(np.imag(Eyfield), cmap='RdBu')
    plt.title('Ey')
    plt.show()
    
    testE = np.zeros(2*Mx*My, dtype=np.complex128)
    testE[::2] = Exfield[designMask]
    testE[1::2] = Eyfield[designMask]
    testJ = (k0/1j/1) * Gddinv @ testE
    testJx = np.reshape(testJ[::2], (Mx,My))
    testJy = np.reshape(testJ[1::2], (Mx,My))
    
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



"""
following code are done with an interpolation step where we put the Ex and Ey fields onto the same grid points
these methods are retained mainly for possible convenience; use of the Yee versions is recommended
"""
def interp_Ex_onto_Ey_grid(Exfield):
    """
    interpolate Exfield onto the Ey Yee positions
    """
    Nx,Ny = Exfield.shape
    interp_Exfield = np.zeros((Nx,Ny), dtype=np.complex128)
    for cx in range(Nx):
        for cy in range(Ny):
            if cx>0:
                prevx = cx-1
            else:
                prevx = Nx-1
            if cy>0:
                prevy = cy-1
            else:
                prevy = Ny-1
            interp_Exfield[cx,cy] = 0.25*(Exfield[cx,cy]+Exfield[cx,prevy]+Exfield[prevx,cy]+Exfield[prevx,prevy])
    return interp_Exfield

def interp_Ey_onto_Ex_grid(Eyfield):
    """
    interpolate Eyfield onto the Ex Yee positions
    """
    Nx,Ny = Eyfield.shape
    interp_Eyfield = np.zeros((Nx,Ny), dtype=np.complex128)
    for cx in range(Nx):
        for cy in range(Ny):
            if cx<Nx-1:
                nextx = cx+1
            else:
                nextx = 0
            if cy<Ny-1:
                nexty = cy+1
            else:
                nexty = 0
            interp_Eyfield[cx,cy] = 0.25*(Eyfield[cx,cy]+Eyfield[cx,nexty]+Eyfield[nextx,cy]+Eyfield[nextx,nexty])
    return interp_Eyfield


def get_TE_GreenFcn(wvlgth, Gx,Gy, Npmlx, Npmly, dx,dy):
    """
    generate Green's function of a domain with shape (Gx,Gy), with a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    gpwx = int(1.0/dx)
    gpwy = int(1.0/dy)
    Nx = 2*Gx-1 + gpwx//2 + 2*Npmlx
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1)
    x_Eyfield = interp_Ey_onto_Ex_grid(x_Eyfield)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2)
    y_Exfield = interp_Ex_onto_Ey_grid(y_Exfield)
    
    numCoord = Gx*Gy
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    
    for ix in range(Gx):
        for iy in range(Gy):
            x_Ex = x_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            x_Ey = x_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ex = y_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ey = y_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            
            xyind = ix*Gy + iy
            G[:numCoord,xyind] = x_Ex.flatten()
            G[numCoord:,xyind] = x_Ey.flatten()
            G[:numCoord,xyind+numCoord] = y_Ex.flatten()
            G[numCoord:,xyind+numCoord] = y_Ey.flatten()
    
    eta = 1.0 #dimensionless units
    Gfac = -1j*(2*np.pi/wvlgth)/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G



def get_periodicY_TE_GreenFcn(wvlgth, Ny, Gx,Gy, dx,dy, Npmlx, Npmlsep=None):
    """
    generate Green's function of a domain with shape (Gx,Gy), with a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    if Npmlsep is None:
        gpw = int(1.0/dx)
        Npmlsep = gpw//2
    Nx = 2*Gx-1 + 2*(Npmlx + Npmlsep)
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, 0, dx, dy, cx, cy, 1)
    x_Eyfield = interp_Ey_onto_Ex_grid(x_Eyfield)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, 0, dx, dy, cx, cy, 2)
    y_Exfield = interp_Ex_onto_Ey_grid(y_Exfield)
    
    x_Exfield = np.concatenate((x_Exfield,x_Exfield,x_Exfield), axis=1)
    x_Eyfield = np.concatenate((x_Eyfield,x_Eyfield,x_Eyfield), axis=1)
    y_Exfield = np.concatenate((y_Exfield,y_Exfield,y_Exfield), axis=1)
    y_Eyfield = np.concatenate((y_Eyfield,y_Eyfield,y_Eyfield), axis=1)
    cy += Ny #stack up 3 unitcells in Y direction for easy window sliding
    
    numCoord = Gx*Gy
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    
    for ix in range(Gx):
        for iy in range(Gy):
            x_Ex = x_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            x_Ey = x_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ex = y_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ey = y_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            
            xyind = ix*Gy + iy
            G[:numCoord,xyind] = x_Ex.flatten()
            G[numCoord:,xyind] = x_Ey.flatten()
            G[:numCoord,xyind+numCoord] = y_Ex.flatten()
            G[numCoord:,xyind+numCoord] = y_Ey.flatten()
    
    eta = 1.0 #dimensionless units
    Gfac = -1j*(2*np.pi/wvlgth)/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G



def get_periodicY_TE_masked_GreenFcn(wvlgth, Ny, Gx,Gy,Gmask, dx,dy, Npmlx, Npmlsep=None):
    """
    generate Green's function of a domain with shape specified by 2D boolean Gmask over a domain of shape (Gx,Gy), within a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    if Npmlsep is None:
        gpw = int(1.0/dx)
        Npmlsep = gpw//2
    Nx = 2*Gx-1 + 2*(Npmlx+Npmlsep)
    cx = Nx//2
    cy = Ny//2
    
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, 0, dx, dy, cx, cy, 1)
    x_Eyfield = interp_Ey_onto_Ex_grid(x_Eyfield)
    
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, 0, dx, dy, cx, cy, 2)
    y_Exfield = interp_Ex_onto_Ey_grid(y_Exfield)
    
    x_Exfield = np.concatenate((x_Exfield,x_Exfield,x_Exfield), axis=1)
    x_Eyfield = np.concatenate((x_Eyfield,x_Eyfield,x_Eyfield), axis=1)
    y_Exfield = np.concatenate((y_Exfield,y_Exfield,y_Exfield), axis=1)
    y_Eyfield = np.concatenate((y_Eyfield,y_Eyfield,y_Eyfield), axis=1)
    cy += Ny #stack up 3 unitcells in Y direction for easy window sliding
    
    numCoord = np.sum(Gmask)
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    
    G_idx = np.argwhere(Gmask)
    
    for i,idx in enumerate(G_idx):
        ulx = cx - idx[0]
        uly = cy - idx[1]
        
        x_Ex = x_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        x_Ey = x_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ex = y_Exfield[ulx:ulx+Gx,uly:uly+Gy]
        y_Ey = y_Eyfield[ulx:ulx+Gx,uly:uly+Gy]
        
        G[:numCoord,i] = x_Ex[Gmask]
        G[numCoord:,i] = x_Ey[Gmask]
        G[:numCoord,i+numCoord] = y_Ex[Gmask]
        G[numCoord:,i+numCoord] = y_Ey[Gmask]
    
    eta = 1.0 #dimensionless units
    Gfac = -1j*(2*np.pi/wvlgth)/eta
    G *= Gfac
    print('check G reciprocity', np.linalg.norm(G-G.T))
    return G


def get_Yee_TE_div_mask(Nx, Ny, dx, dy, divmask):
    """
    get the divergence operator for gridpoints set by boolean divmask
    basis ordering: (0,0,Ex)...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """

    points = np.argwhere(divmask)
    div = np.zeros((points.shape[0],2*Nx*Ny), dtype=np.complex128)
    invdx = 1.0/dx
    invdy = 1.0/dy
    for i in range(points.shape[0]):
        x = points[i,0]
        y = points[i,1]

        #allow for periodic boundary conditions
        if x>0:
            xm1 = x-1
        else:
            xm1 = Nx-1

        if y<Ny-1:
            yp1 = y+1
        else:
            yp1 = 0

        div[i,x*Ny+y] = invdx #+x
        div[i,xm1*Ny+y] = -invdx #-x
        div[i,Nx*Ny+x*Ny+yp1] = invdy #+y
        div[i,Nx*Ny+x*Ny+y] = -invdy #-y

    return div

        
