#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 14:32:29 2021

@author: pengning
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt

def make_Dxc(dL, shape, bloch_x=0.0):
    """ center differences derivative in x """
    Nx, Ny = shape
    fx_phasor = np.exp(1j*bloch_x)
    bx_phasor = np.exp(-1j*bloch_x)
    
    Dxc = sp.diags([fx_phasor,-1,1,-bx_phasor], [-Nx+1,-1,1,Nx-1], shape=(Nx,Nx), dtype=np.complex128)
    Dxc = (1 / (2*dL)) * sp.kron(Dxc, sp.eye(Ny), format="csc")
    return Dxc

def make_Dxf(dL, shape, bloch_x=0.0):
    """ Forward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxf = sp.diags([-1, 1, phasor_x], [0, 1, -Nx+1], shape=(Nx, Nx), dtype=np.complex128)
    Dxf = 1 / dL * sp.kron(Dxf, sp.eye(Ny), format="csc")
    return Dxf

def make_Dxb(dL, shape, bloch_x=0.0):
    """ Backward derivative in x """
    Nx, Ny = shape
    phasor_x = np.exp(1j * bloch_x)
    Dxb = sp.diags([1, -1, -np.conj(phasor_x)], [0, -1, Nx-1], shape=(Nx, Nx), dtype=np.complex128)
    Dxb = 1 / dL * sp.kron(Dxb, sp.eye(Ny), format="csc")
    return Dxb

def make_Dyc(dL, shape, bloch_y=0.0):
    """ center differences derivative in y """
    Nx, Ny = shape
    fy_phasor = np.exp(1j*bloch_y)
    by_phasor = np.exp(-1j*bloch_y)
    
    Dyc = sp.diags([fy_phasor,-1,1,-by_phasor], [-Ny+1,-1,1,Ny-1], shape=(Ny,Ny), dtype=np.complex128)
    Dyc = (1 / (2*dL)) * sp.kron(sp.eye(Nx), Dyc, format="csc")
    return Dyc

def make_Dyf(dL, shape, bloch_y=0.0):
    """ Forward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyf = sp.diags([-1, 1, phasor_y], [0, 1, -Ny+1], shape=(Ny, Ny))
    Dyf = 1 / dL * sp.kron(sp.eye(Nx), Dyf, format="csc")
    return Dyf

def make_Dyb(dL, shape, bloch_y=0.0):
    """ Backward derivative in y """
    Nx, Ny = shape
    phasor_y = np.exp(1j * bloch_y)
    Dyb = sp.diags([1, -1, -np.conj(phasor_y)], [0, -1, Ny-1], shape=(Ny, Ny))
    Dyb = 1 / dL * sp.kron(sp.eye(Nx), Dyb, format="csc")
    return Dyb


C_0 = 1.0 #dimensionless units
EPSILON_0 = 1.0
ETA_0 = 1.0

#########################PML###################################################
def create_S_matrices(omega, shape, npml, dL):
    """ Makes the 'S-matrices'.  When dotted with derivative matrices, they add PML """

    # strip out some information needed
    Nx, Ny = shape
    N = Nx * Ny
    Nx_pml, Ny_pml = npml    

    # Create the sfactor in each direction and for 'f' and 'b'
    s_vector_x_f = create_sfactor('f', omega, dL, Nx, Nx_pml)
    s_vector_x_b = create_sfactor('b', omega, dL, Nx, Nx_pml)
    s_vector_y_f = create_sfactor('f', omega, dL, Ny, Ny_pml)
    s_vector_y_b = create_sfactor('b', omega, dL, Ny, Ny_pml)

    # Fill the 2D space with layers of appropriate s-factors
    Sx_f_2D = np.zeros(shape, dtype=np.complex128)
    Sx_b_2D = np.zeros(shape, dtype=np.complex128)
    Sy_f_2D = np.zeros(shape, dtype=np.complex128)
    Sy_b_2D = np.zeros(shape, dtype=np.complex128)

    # insert the cross sections into the S-grids (could be done more elegantly)
    for i in range(0, Ny):
        Sx_f_2D[:, i] = 1 / s_vector_x_f
        Sx_b_2D[:, i] = 1 / s_vector_x_b
    for i in range(0, Nx):
        Sy_f_2D[i, :] = 1 / s_vector_y_f
        Sy_b_2D[i, :] = 1 / s_vector_y_b

    # Reshape the 2D s-factors into a 1D s-vecay
    Sx_f_vec = Sx_f_2D.flatten()
    Sx_b_vec = Sx_b_2D.flatten()
    Sy_f_vec = Sy_f_2D.flatten()
    Sy_b_vec = Sy_b_2D.flatten()

    # Construct the 1D total s-vecay into a diagonal matrix
    Sx_f = sp.spdiags(Sx_f_vec, 0, N, N, format="csc")
    Sx_b = sp.spdiags(Sx_b_vec, 0, N, N, format="csc")
    Sy_f = sp.spdiags(Sy_f_vec, 0, N, N, format="csc")
    Sy_b = sp.spdiags(Sy_b_vec, 0, N, N, format="csc")

    return Sx_f, Sx_b, Sy_f, Sy_b

def create_sfactor(dir, omega, dL, N, N_pml):
    """ creates the S-factor cross section needed in the S-matrices """

    #  for no PNL, this should just be zero
    if N_pml == 0:
        return np.ones(N, dtype=np.complex128)

    # otherwise, get different profiles for forward and reverse derivative matrices
    dw = N_pml * dL
    if dir == 'f':
        return create_sfactor_f(omega, dL, N, N_pml, dw)
    elif dir == 'b':
        return create_sfactor_b(omega, dL, N, N_pml, dw)
    else:
        raise ValueError("Dir value {} not recognized".format(dir))

def create_sfactor_f(omega, dL, N, N_pml, dw):
    """ S-factor profile for forward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 0.5), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 0.5), dw, omega)
    return sfactor_array

def create_sfactor_b(omega, dL, N, N_pml, dw):
    """ S-factor profile for backward derivative matrix """
    sfactor_array = np.ones(N, dtype=np.complex128)
    for i in range(N):
        if i <= N_pml:
            sfactor_array[i] = s_value(dL * (N_pml - i + 1), dw, omega)
        elif i > N - N_pml:
            sfactor_array[i] = s_value(dL * (i - (N - N_pml) - 1), dw, omega)
    return sfactor_array

def sig_w(l, dw, m=3, lnR=-30):
    """ Fictional conductivity, note that these values might need tuning """
    sig_max = -(m + 1) * lnR / (2 * ETA_0 * dw)
    return sig_max * (l / dw)**m

def s_value(l, dw, omega):
    """ S-value to use in the S-matrices """
    return 1 + 1j * sig_w(l, dw) / (omega * EPSILON_0)



#Testing

def get_TM_MaxwellOp(wvlgth, dL, Nx, Ny, Npml, bloch_x=0.0, bloch_y=0.0, Qabs=np.inf):
    """
    uniform grid 2D scalar E field Maxwell operator
    Npml can be both an int or a tuple
    Parameters
    ----------
    wvlgth : float
        wavelength of interest in units of 1.
    dL : float
        finite difference grid pixel size. 
    Nx : int
        number of pixels along the x direction.
    Ny : int
        number of pixels along the y direction.
    Npml : int or tuple
        number of pixels in the PML region (part of Nx and Ny).
        if Npml is int it will be promoted to tuple (Npml,Npml).
    bloch_x : float, optional
        x-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    bloch_y : float, optional
        y-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    Qabs : float, optional
        Q parameter specifying bandwidth of sources. The default is np.inf.

    Returns
    -------
    M : sparse complex matrix
        Maxwell operator in sparse matrix format.

    """

    shape = (Nx,Ny)
    if type(Npml)==int:
        Npml = (Npml,Npml)
    
    Dxf = make_Dxf(dL, shape, bloch_x=bloch_x)
    Dxb = make_Dxb(dL, shape, bloch_x=bloch_x)
    Dyf = make_Dyf(dL, shape, bloch_y=bloch_y)
    Dyb = make_Dyb(dL, shape, bloch_y=bloch_y)

    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs)
    
    Sxf, Sxb, Syf, Syb = create_S_matrices(omega, shape, Npml, dL)
    
    #dress the derivative functions with pml
    Dxf = Sxf @ Dxf
    Dxb = Sxb @ Dxb

    Dyf = Syf @ Dyf
    Dyb = Syb @ Dyb

    M = -Dxf @ Dxb - Dyf @ Dyb - EPSILON_0*omega**2 * sp.eye(Nx*Ny)
    return M


def get_diagM_from_chigrid(omega, chigrid):
    Nx,Ny = chigrid.shape
    return -sp.diags(chigrid.flatten() * omega**2)

def get_TM_dipole_field(wvlgth, dL, Nx, Ny, cx, cy, Npml, bloch_x=0.0, bloch_y=0.0, Qabs=np.inf, chigrid=None):
    """
    get the field of a TM dipole source at position (cx,cy) in a grid of size (Nx,Ny)
    and material distribution given by chigrid.
    
    Parameters
    ----------
    wvlgth : float
        wavelength of interest.
    dL : float
        size of a single pixel of the finite difference grid.
    Nx : int
        Number of pixels along the x direciton.
    Ny : int
        Number of pixels along the y direction.
    cx : int
        x-coordinate of the dipole source.
    cy : int
        y-coordinate of the dipole source.
    Npml : int
        number of grid points in the PML.
    loch_x : float, optional
        x-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    bloch_y : float, optional
        y-direction phase shift associated with the periodic boundary condtions. The default is 0.0.
    Qabs : float, optional
        Q parameter specifying bandwidth of sources. The default is np.inf.
    chigrid : 2D numpy complex array, optional
        spatial distribution of material susceptibility. The default is None, corresponding to vacuum.

    Returns
    -------
    Ez : 2D numpy complex array
        Field of the dipole source.

    """
    shape = (Nx,Ny)
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs)
    
    M = get_TM_MaxwellOp(wvlgth, dL, Nx,Ny, Npml, bloch_x=bloch_x, bloch_y=bloch_y, Qabs=Qabs)

    if not (chigrid is None):
        M += get_diagM_from_chigrid(omega, chigrid)
    
    sourcegrid = np.zeros((Nx,Ny), dtype=np.complex)
    sourcegrid[cx,cy]=1.0 / (dL**2)
    RHS = 1j*omega*sourcegrid.flatten()
    
    Ez = spla.spsolve(M, RHS)
    
    Ez = np.reshape(Ez, shape)
    return Ez

    
def plot_TM_dipole_field(wvlgth, dL, Nx, Ny, cx, cy, Npml, probex, probey, Qabs=np.inf):

    Ez = get_TM_dipole_field(wvlgth, dL, Nx, Ny, cx, cy, Npml, Qabs=Qabs)
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Ez), cmap='RdBu')
    ax2.imshow(np.imag(Ez), cmap='RdBu')
    plt.show()

    Ezabs = np.real(Ez * np.conj(Ez))
    plt.imshow(Ezabs)
    plt.show()
    
    print('Ez at probe', Ez[probex,probey])


def get_TM_G_od(wvlgth, dL, nonpmlNx, nonpmlNy, Npml, design_mask, observe_mask, bloch_x=0.0, bloch_y=0.0, Qabs=np.inf):
    """
    calculate Green's function from design_mask region to observe_mask region
    from vacuum dipole field using a sliding window to exploit translation symmetry
    """
    if type(Npml)==int:
        Npml = (Npml,Npml)
    Npmlx, Npmly = Npml
    bigNx = 2*nonpmlNx-1 + 2*Npmlx
    bigNy = 2*nonpmlNy-1 + 2*Npmly
    bigcx = Npmlx+nonpmlNx-1
    bigcy = Npmly+nonpmlNy-1
    omega = (2*np.pi*C_0/wvlgth) * (1 + 1j/2/Qabs)
    
    A = get_TM_MaxwellOp(wvlgth, dL, bigNx, bigNy, Npml, bloch_x=bloch_x, bloch_y=bloch_y, Qabs=Qabs)
    sourcegrid = np.zeros((bigNx,bigNy), dtype=np.complex)
    sourcegrid[bigcx,bigcy] = 1.0 / dL
    RHS = 1j*omega*sourcegrid.flatten()

    Ezfield = spla.spsolve(A, RHS)
    Ezfield = np.reshape(Ezfield, (bigNx,bigNy))
    design_idx = np.argwhere(design_mask)
    G_od = np.zeros((np.sum(observe_mask), design_idx.shape[0]), dtype=np.complex)
    for i,idx in enumerate(design_idx):
        ulx = bigcx - idx[0]
        uly = bigcy - idx[1] #big grid coordinates of upper-left corner of small grid situated on top of the big grid to get zdipole field at (idx[0],idx[1]) in small grid coordinates
        G_od[:,i] = (Ezfield[ulx:ulx+nonpmlNx,uly:uly+nonpmlNy])[observe_mask]

    k = omega / C_0
    G_od *= dL * (-1j*k/ETA_0) #scale Ezfield to get true representation of Green's function by our definition; E = (iZ/k) * G @ J
    return G_od
    

def get_Gddinv(wvlgth, dL, Nx,Ny, Npml, designMask, bloch_x=0.0, bloch_y=0.0, Qabs=np.inf):
    #use Woodbury inversion to get G_dd^-1 from Maxwell operator
    
    M = get_TM_MaxwellOp(wvlgth, dL, Nx,Ny, Npml, bloch_x=bloch_x, bloch_y=bloch_y, Qabs=Qabs)
    #M = M.tocsc()
    print('type of M', type(M))
    designInd = np.nonzero(designMask.flatten())[0]
    backgroundInd = np.nonzero(np.logical_not(designMask.flatten()))[0]
    
    A = (M[:,backgroundInd])[backgroundInd,:]
    B = (M[:,designInd])[backgroundInd,:]
    C = (M[:,backgroundInd])[designInd,:]
    D = (M[designInd,:])[:,designInd]
    
    AinvB = spla.spsolve(A, B)
    print(type(AinvB))
    
    k = (2*np.pi/wvlgth) * (1 + 1j/2/Qabs)
    MU_0 = 1.0
    Gfac = MU_0 / k**2
    return (D - (C @ AinvB))*Gfac, M


def test_Gdd_inv():
    wvlgth=1.0
    dL = 0.05
    Nx = 120
    Ny = 120
    Npmlx = 20
    Npmly = 20
    Npml = (Npmlx,Npmly)
    Qabs = 100
    
    bloch_x=0.0
    bloch_y=0.0
    
    #Qabs = 10
    Mx = 20
    My = 20
    Mx0 = (Nx-Mx)//2
    My0 = (Ny-My)//2
    designMask = np.zeros((Nx,Ny), dtype=np.bool)
    designMask[Mx0:Mx0+Mx,My0:My0+My] = True
    
    Gddinv, M = get_Gddinv(wvlgth, dL, Nx, Ny, Npml, designMask, bloch_x=bloch_x, bloch_y=bloch_y, Qabs=Qabs)
    
    #test definiteness of Gdd
    Gdd = get_TM_G_od(wvlgth, dL, Mx, My, Npmlx, np.ones((Mx,My), dtype=np.bool), np.ones((Mx,My), dtype=np.bool), Qabs=Qabs)
    AsymG = (Gdd - Gdd.conj().T) / 2j
    eigw = np.linalg.eigvalsh(AsymG)
    print('check AsymG definiteness', eigw[0], eigw[-1])
    #add a phase projection
    Gdd_P = Gdd * (1-1j/2/Qabs)
    AsymG_P = (Gdd_P - Gdd_P.conj().T) / 2j
    eigw = np.linalg.eigvalsh(AsymG_P)
    print('check AsymG_P definiteness', eigw[0], eigw[-1])
    
    
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    cx = Mx0 + Mx//2
    cy = My0 + My//2
    sourcegrid = np.zeros((Nx,Ny), dtype=np.complex)
    sourcegrid[cx,cy]=1.0 / (dL**2)
    RHS = 1j*omega*sourcegrid.flatten()
    print('Ez of point source')
    Ez = spla.spsolve(M,RHS)
    Ez = np.reshape(Ez, (Nx,Ny))
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Ez), cmap='RdBu')
    ax2.imshow(np.imag(Ez), cmap='RdBu')
    plt.show()
    
    print('Gdd^{-1} applied upon Ez restricted to design domain')
    designEz = Ez[designMask]
    testSource = -1j*EPSILON_0*omega*Gddinv.dot(designEz)
    testSource = testSource.reshape((Mx,My))
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(testSource), cmap='RdBu')
    ax2.imshow(np.imag(testSource), cmap='RdBu')
    plt.show()
    print(np.max(np.abs(np.real(testSource))))
    print(np.max(np.abs(np.imag(testSource))))
    return Gddinv


    
