#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:11:19 2020

@author: pengning
"""

import numpy as np
import scipy.special as sp
from scipy.integrate import dblquad
import matplotlib.pyplot as plt

#EPSILON_0 = 8.85418782e-12        # vacuum permittivity
#MU_0 = 1.25663706e-6              # vacuum permeability
#C_0 = 1 / np.sqrt(EPSILON_0 * MU_0)  # speed of light in vacuum

EPSILON_0 = 1.0        # vacuum permittivity
MU_0 = 1.0             # vacuum permeability
C_0 = 1.0              # speed of light in vacuum

def TMdipole_2D_SI(wvlgth, rho, amp=1.0):
    k0 = 2*np.pi/wvlgth
    omega = C_0 * k0
    return -0.25*omega*MU_0*amp * sp.hankel1(0, k0*rho)


def TEydipole_2D_SI(wvlgth, x, y, amp=1.0):
    k0 = 2*np.pi/wvlgth
    rho = np.hypot(x,y)
    sinphi = -x / rho
    cosphi = y / rho
    unitrho = np.array([x,y]) / rho
    unitphi = np.array([-y,x]) / rho
    
    E_rho = -(amp/(4*C_0*EPSILON_0*rho)) * cosphi * sp.hankel1(1,k0*rho)
    E_phi = (amp*k0/(4*C_0*EPSILON_0)) * sinphi * sp.h1vp(1,k0*rho)
    
    return E_rho*unitrho + E_phi*unitphi


def TExdipole_2D_SI(wvlgth, x, y, amp=1.0):
    E_rotated = TEydipole_2D_SI(wvlgth, -y, x, amp=amp)
    return np.array([E_rotated[1], -E_rotated[0]])


def phasor_rotate_fac(c1,c2):
    #returns unit norm cplx factor that rotates c1 to be of same phase as c2
    c2 /= np.sqrt(np.real(c2*np.conj(c2)))
    c1 /= np.sqrt(np.real(c1*np.conj(c1)))
    return c2 / c1


def integrate_around_center(func, dl, eps=1e-2):
    lx = -dl/2; rx = dl/2
    ly = -dl/2; ry = dl/2
    lex = -dl*eps/2; rex = dl*eps/2
    ley = -dl*eps/2; rey = dl*eps/2
    
    int1, err = dblquad(func, lx, rx, lambda x: rey, lambda x: ry)
    print(err, int1)
    int2, err = dblquad(func, lx, rx, lambda x: ly, lambda x: ley)
    print(err, int2)
    int3, err = dblquad(func, lx, lex, lambda x: ley, lambda x: rey)
    print(err, int3)
    int4, err = dblquad(func, rex, rx, lambda x: ley, lambda x: rey)
    print(err, int4)
    return int1+int2+int3+int4


def TMdipole_SI_avg_center_field(wvlgth, dl, amp=1.0, eps=1e-2):
    Ez_real = lambda x,y: np.real(TMdipole_2D_SI(wvlgth, np.hypot(x,y), amp=amp))
    Ez_imag = lambda x,y: np.imag(TMdipole_2D_SI(wvlgth, np.hypot(x,y), amp=amp))
    
    int_Ez_real = integrate_around_center(Ez_real, dl, eps=eps)
    int_Ez_imag = integrate_around_center(Ez_imag, dl, eps=eps)
    
    denom = dl**2 - (dl*eps)**2
    return (int_Ez_real+1j*int_Ez_imag)/denom

def get_TMdipole_SI_field_grid(wvlgth, dl, Nx, Ny, cx, cy, amp=1.0):
    Ezfield = np.zeros((Nx,Ny), dtype=np.complex)
    for i in range(Nx):
        for j in range(Ny):
            if i==cx and j==cy:
                Ezfield[i,j] = TMdipole_SI_avg_center_field(wvlgth, dl, amp=amp)
            else:
                Ezfield[i,j] = TMdipole_2D_SI(wvlgth, np.hypot((i-cx)*dl, (j-cy)*dl), amp=amp)
    return Ezfield


def TEdipole_SI_avg_center_field(pol, wvlgth, dl, amp=1.0, eps=1e-2):
    if pol=='x':
        dipolefunc = TExdipole_2D_SI
    elif pol=='y':
        dipolefunc = TEydipole_2D_SI
    else:
        raise ValueError('pol must be x or y')
    Ex_real = lambda x,y: np.real(dipolefunc(wvlgth, x,y, amp=amp)[0])
    Ex_imag = lambda x,y: np.imag(dipolefunc(wvlgth, x,y, amp=amp)[0])
    Ey_real = lambda x,y: np.real(dipolefunc(wvlgth, x,y, amp=amp)[1])
    Ey_imag = lambda x,y: np.imag(dipolefunc(wvlgth, x,y, amp=amp)[1])
    
    int_Ex_real = integrate_around_center(Ex_real, dl, eps=eps)
    int_Ex_imag = integrate_around_center(Ex_imag, dl, eps=eps)
    int_Ey_real = integrate_around_center(Ey_real, dl, eps=eps)
    int_Ey_imag = integrate_around_center(Ey_imag, dl, eps=eps)
    
    denom = dl**2 - (dl*eps)**2
    return np.array([(int_Ex_real+1j*int_Ex_imag)/denom, (int_Ey_real+1j*int_Ey_imag)/denom])


def get_TEdipole_SI_field_grid(pol, wvlgth, dl, Nx, Ny, cx, cy, amp=1.0):
    if pol=='x':
        dipolefunc = TExdipole_2D_SI
    elif pol=='y':
        dipolefunc = TEydipole_2D_SI
    else:
        raise ValueError('pol must be x or y')
    
    Exfield = np.zeros((Nx,Ny), dtype=np.complex)
    Eyfield = np.zeros((Nx,Ny), dtype=np.complex)
    
    for i in range(Nx):
        for j in range(Ny):
            if i==cx and j==cy:
                Efield = TEdipole_SI_avg_center_field(pol, wvlgth, dl, amp=amp)
            else:
                Efield = dipolefunc(wvlgth, (i-cx)*dl, (j-cy)*dl, amp=amp)
            Exfield[i,j] = Efield[0]
            Eyfield[i,j] = Efield[1]
    
    return Exfield, Eyfield


def get_TEdipole_SI_periodicy_field_grid(pol, wvlgth, dl, Nx, Ny, cx, cy, amp=1.0, tol=1e-4, sumdelta=10):
    if pol=='x':
        dipolefunc = TExdipole_2D_SI
    elif pol=='y':
        dipolefunc = TEydipole_2D_SI
    else:
        raise ValueError('pol must be x or y')
        
    Exfield = np.zeros((Nx,Ny), dtype=np.complex)
    Eyfield = np.zeros((Nx,Ny), dtype=np.complex)
    
    for i in range(Nx):
        print('i', i)
        for j in range(Ny):
            if i==cx and j==cy:
                Efield = TEdipole_SI_avg_center_field(pol, wvlgth, dl, amp=amp)
            else:
                Efield = dipolefunc(wvlgth, (i-cx)*dl, (j-cy)*dl, amp=amp)
            #sum up contributions from farther out unit cells until convergence
            
            n = 1
            nlim=sumdelta
            deltaE = np.zeros(2, dtype=np.complex)
            while n<nlim:
                print('n', n)
                deltaE += dipolefunc(wvlgth, (i-cx)*dl, (j+n*Ny-cy)*dl, amp=amp)
                deltaE += dipolefunc(wvlgth, (i-cx)*dl, (j-n*Ny-cy)*dl, amp=amp)
                n += 1
                if n==nlim:
                    print('Efield', Efield)
                    print('deltaE', deltaE)
                    if np.linalg.norm(deltaE)>np.linalg.norm(Efield)*tol:
                        nlim += sumdelta
                    Efield += deltaE
                    deltaE[:] = 0
            
            Exfield[i,j] = Efield[0]
            Eyfield[i,j] = Efield[1]
    
    return Exfield, Eyfield


def test_periodic_dipole_field(pol, wvlgth, dl, Nx, Ny, cx, cy, amp=1.0, tol=1e-4, sumdelta=10):
    Exfield, Eyfield = get_TEdipole_SI_periodicy_field_grid(pol, wvlgth, dl, Nx, Ny, cx, cy, amp=amp, tol=tol, sumdelta=sumdelta)
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Exfield), cmap='RdBu')
    ax2.imshow(np.imag(Exfield), cmap='RdBu')
    plt.show()
    
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Eyfield), cmap='RdBu')
    ax2.imshow(np.imag(Eyfield), cmap='RdBu')
    plt.show()
    
    
def plot_field(fieldname):
    Exfield = np.load(fieldname+'_Exfield.npy')
    Eyfield = np.load(fieldname+'_Eyfield.npy')
    
    realExmax = np.max(np.real(Exfield))
    realExmin = np.min(np.real(Exfield))
    print('realExmax', realExmax, 'realExmin', realExmin)
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Exfield), cmap='RdBu', vmin=realExmin, vmax=realExmax)
    ax2.imshow(np.imag(Exfield), cmap='RdBu', vmin=realExmin, vmax=realExmax)
    plt.show()
    
    realEymax = np.max(np.real(Eyfield))
    realEymin = np.min(np.real(Eyfield))
    print('realEymax', realEymax, 'realEymin', realEymin)
    fig, (ax1,ax2) = plt.subplots(ncols=2)
    ax1.imshow(np.real(Eyfield), cmap='RdBu', vmin=realEymin, vmax=realEymax)
    ax2.imshow(np.imag(Eyfield), cmap='RdBu', vmin=realEymin, vmax=realEymax)
    plt.show()
    
    
def check_field_reciprocity(fieldname, scx, scy, probex, probey):
    Exfield = np.load(fieldname+'_Exfield.npy')
    Eyfield = np.load(fieldname+'_Eyfield.npy')
    
    print('Ex at probe', Exfield[probex,probey])
    print('Ey at probe', Eyfield[probex,probey])
    
    print('recip Ex at source', Exfield[2*scx-probex,2*scy-probey])
    print('recip Ey at source', Eyfield[2*scx-probex,2*scy-probey])
    
    