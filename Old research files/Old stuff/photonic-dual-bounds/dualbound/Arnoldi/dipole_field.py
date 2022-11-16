#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:29:49 2019

@author: pengning
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import mpmath
from mpmath import mp
axisfont = {'fontsize':'18'}

#implement function that returns dipole field at given point

#dipole is polarized in z direction, located at (xd,yd,zd)
#the field at point (xp,yp,zp) is outputted as a numpy array with 3 elements
#k is size of wavevector; dipole field not scale invariant
def zdipole_field(k,xd,yd,zd,xp,yp,zp):
    x = xp-xd; y=yp-yd; z=zp-zd #apply formula with dipole at origin 
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    eikr = np.exp(1j*kr)
    field = np.array([0.0j,0.0j,0.0j]) #(Ex,Ey,Ez) at (xp,yp,zp)
    
    field[0] = x*z*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5)
    field[1] = y*z*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5)
    field[2] = eikr/(4.0*np.pi*r)
    field[2] += (1j*kr-1.0)*eikr/(4.0*np.pi*k**2*r**3) + z**2*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5)

    return field

def zdipole_field_array(k,xd,yd,zd,xp,yp,zp):
    xp = np.atleast_3d(xp); yp = np.atleast_3d(yp); zp = np.atleast_3d(zp)
    x = xp-xd; y=yp-yd; z=zp-zd #apply formula with dipole at origin 
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    eikr = np.exp(1j*kr)
    #field = np.array([0.0j,0.0j,0.0j]) #(Ex,Ey,Ez) at (xp,yp,zp)
    field = np.zeros(xp.shape + (3,),dtype=complex)
    field[...,0] = x*z*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5) #Ex
    field[...,1] = y*z*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5) #Ey
    field[...,2] = eikr/(4.0*np.pi*r) #Ez
    field[...,2] += (1j*kr-1.0)*eikr/(4.0*np.pi*k**2*r**3) + z**2*(-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5)

    return field

#plot 3D vector part of dipole field, in box bounded by lx,ux,...... with dipole location (xd,yd,zd)
def plot_dipole_field_3D(k,xd,yd,zd,lx,ux,ly,uy,lz,uz,normal=False):
    density = 4
    xg = np.linspace(lx,ux,density); yg = np.linspace(ly,uy,density); zg = np.linspace(lz,uz,density)
    [xg,yg,zg] = np.meshgrid(xg,yg,zg,indexing='ij')
    
    field = zdipole_field_array(k,xd,yd,zd,xg,yg,zg)
    fig, ax = plt.subplots(1,1)
    ax = fig.gca(projection='3d')
    ax.quiver3D(xg,yg,zg,np.real(field[:,:,:,0]),np.real(field[:,:,:,1]),np.real(field[:,:,:,2]),normalize=normal)
    plt.show()

#plot a 2D streamplot with dipole in center so there are no out-of-plane fields
def plot_dipole_field_2Dstream(k,ly,uy,lz,uz):
    dense = 50
    y1d = np.linspace(ly,uy,dense); z1d = np.linspace(lz,uz,dense)
    [xg,yg,zg] = np.meshgrid([0.0],y1d,z1d,indexing='ij') #note that meshgrid does xy indexing by default
    field = zdipole_field_array(k,0,0,0,xg,yg,zg)
    
    plt.figure()
    #all the transposing to deal with the issue that streamplot is hardwired with xy indexing, while we use ij indexing
    plt.streamplot(np.transpose(yg[0,:,:]),np.transpose(zg[0,:,:]),np.real(np.transpose(field[0,:,:,1])),np.real(np.transpose(field[0,:,:,2])),density=2.5)
    plt.xlim(ly,uy)
    plt.ylim(lz,uz)
    plt.show()

def xdipole_field(k,xd,yd,zd,xp,yp,zp):
    x = xp-xd; y = yp-yd; z = zp-zd
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    eikr = np.exp(1j*kr)
    field = np.array([0.0j,0.0j,0.0j]) #(Ex,Ey,Ez) at (xp,yp,zp)
    
    factor = (-kr**2-3j*kr+3.0)*eikr / (4.0*np.pi*k**2*r**5)
    field[2] = x*z*factor
    field[1] = y*x*factor
    field[0] = eikr/(4.0*np.pi*r)
    field[0] += (1j*kr-1.0)*eikr/(4.0*np.pi*k**2*r**3) + x**2*factor

    return field
#################expansion of off origin dipole field in spherical waves########
    
#for zpolarized dipole along z axis only the RgNl0 waves are needed to represent dipole field
#for region above dipole should use U^-, A^- in Kardar paper expression
#normalization depends on given domain and will be treated in the calculation files for each domain, not here


#mpmath based high precision spherical bessel
def mp_spherical_jn(l,z):
    return mpmath.sqrt(mpmath.pi/2.0/z)*mpmath.besselj(l+0.5,z)

def mp_vec_spherical_jn(l,z):
    vecfcn = np.vectorize(mp_spherical_jn)
    return vecfcn(l,z)

def mp_spherical_yn(l,z):
    return mpmath.sqrt(mpmath.pi/2.0/z)*mpmath.bessely(l+0.5,z)

def mp_vec_spherical_yn(l,z):
    vecfcn = np.vectorize(mp_spherical_yn)
    return vecfcn(l,z)

def mp_spherical_hn(l,z):
    return mp_spherical_jn(l,z) + 1j*mp_spherical_yn(l,z)

def mp_spherical_djn(l,z):
    return mpmath.sqrt(mpmath.pi/(2*z)) * (mpmath.besselj(l+0.5,z,1) - mpmath.besselj(l+0.5,z)/(2*z))

def mp_vec_spherical_djn(l,z):
    vecfcn = np.vectorize(mp_spherical_djn)
    return vecfcn(l,z)

def mp_spherical_dyn(l,z):
    return mpmath.sqrt(mpmath.pi/(2*z)) * (mpmath.bessely(l+0.5,z,1) - mpmath.bessely(l+0.5,z)/(2*z))

def mp_vec_spherical_dyn(l,z):
    vecfcn = np.vectorize(mp_spherical_dyn)
    return vecfcn(l,z)


def cplxquad(func, intmin, intmax):
    refunc = lambda x: np.real(func(x))
    imfunc = lambda x: np.imag(func(x))
    int_re, err = integrate.quad(refunc, intmin, intmax)
    int_im, err = integrate.quad(imfunc, intmin, intmax)
    return int_re + 1j*int_im

from scipy import integrate
def check_G_on_spherical_RgM(n,r,R):
    #numerically evaluate the radial integral involved in G dot RgM
    int1, err = integrate.quad(lambda x: x**2 * (mp_spherical_jn(n,x))**2, 0,r)
    int2 = cplxquad(lambda x: x**2*mp_spherical_hn(n,x)*mp_spherical_jn(n,x), r, R)
    
    radial = mp_spherical_hn(n,r)*int1 + mp_spherical_jn(n,r)*int2
    print(radial)
    print(mp_spherical_jn(n,r))
    return radial / mp_spherical_jn(n,r)

def check_G_on_spherical_M(n,r,R):
    #numerically evaluate the radial integral involved in G dot M
    int1 = cplxquad(lambda x: x**2 * mp_spherical_jn(n,x) * mp_spherical_hn(n,x), 0,r)
    int2 = cplxquad(lambda x: x**2*mp_spherical_hn(n,x)**2, r, R)
    
    radial = mp_spherical_hn(n,r)*int1 + mp_spherical_jn(n,r)*int2
    print(radial)
    print(mp_spherical_hn(n,r))
    return radial / mp_spherical_hn(n,r)


def zdipole_field_xy2D_periodic_array(k0,L, xd,yd,zd, xp,yp,zp, sumtol=1e-12):
    #field at coord (xp,yp,zp) of z-polarized dipoles in a square array in the z=czd plane
    #generated by summing over discrete k-vectors; in evanescent region the summand decreases exponentially with increasing kp and abskz
    #it seems that directly summing over individual dipole fields leads to convergence issues due to relatively slow (polynomial) decay of dipole fields
    #order of summation in k-space: concentric squares, 0th square the origin, 1st square infinity-norm radius 1, 2nd square infinity-norm radius 2...
    #since we are using the angular representation for consistency we insist that zp>zd
    """
    if not (xp<L/2 and xp>-L/2 and yp<L/2 and yp>-L/2):
        return 'target point out of Brillouin zone'
    """
    if zp<=zd:
        return 'need zp>zd'
    
    field = mp.zeros(3,1)
    oldfield = mp.zeros(3,1)
    deltak = 2*mp.pi/L
    deltax = xp-xd; deltay = yp-yd; deltaz = zp-zd
    
    i = 1 #label of which square we are on
    
    prefac = -1j/(2 * L**2 * k0**2)
    while True: #termination condition in loop
        #sum over the i-th square
        lkx = -i*deltak; rkx = i*deltak
        for iy in range(-i,i+1):
            ky = iy*deltak
            kpsqr = lkx**2 + ky**2
            kz = mp.sqrt(k0**2-kpsqr)
            lphasefac = mp.expj(lkx*deltax + ky*deltay + kz*deltaz)
            rphasefac = mp.expj(rkx*deltax + ky*deltay + kz*deltaz)
            
            field[0] += prefac*(lkx*lphasefac + rkx*rphasefac)
            field[1] += prefac*(lphasefac+rphasefac)*ky
            field[2] += prefac*(lphasefac+rphasefac)*(-kpsqr/kz)
        
        bky = -i*deltak; uky = i*deltak
        for ix in range(-i+1,i):
            kx = ix*deltak
            kpsqr = kx**2 + bky**2
            kz = mp.sqrt(k0**2-kpsqr)
            bphasefac = mp.expj(kx*deltax + bky*deltay + kz*deltaz)
            uphasefac = mp.expj(kx*deltax + uky*deltay + kz*deltaz)
            
            field[0] += prefac*(bphasefac+uphasefac)*kx
            field[1] += prefac*(bphasefac*bky + uphasefac*uky)
            field[2] += prefac*(bphasefac+uphasefac)*(-kpsqr/kz)
            
        if mp.norm(field-oldfield)<mp.norm(field)*sumtol:
            break
        
        print('i',i)
        #mp.nprint(field)
        oldfield = field.copy()
        i+=1
    
    return field

def zdipole_field_xy2D_periodic_array_from_individuals(k0,L, cxd,cyd,czd, xp,yp,zp, sumtol=1e-3):
    field = zdipole_field(k0, cxd,cyd,czd, xp,yp,zp)
    oldfield = field.copy()
    
    i=1 #we treat i=0 at beginning to avoid double counting
    while True:
        lxd = cxd-i*L; rxd = cxd+i*L
        for iy in range(-i,i+1):
            yd = cyd+iy*L
            field += zdipole_field(k0, lxd,yd,czd, xp,yp,zp) + zdipole_field(k0,rxd,yd,czd, xp,yp,zp)
        
        byd = cyd-i*L; uyd = cyd+i*L
        for ix in range(-i+1,i):
            xd = cxd+ix*L
            field += zdipole_field(k0,xd,byd,czd, xp,yp,zp) + zdipole_field(k0,xd,uyd,czd, xp,yp,zp)
        
        if np.linalg.norm(field-oldfield)<np.linalg.norm(field)*sumtol:
            break
        
        print('i',i,'field',field)
        oldfield = field.copy()
        i+=1
    return field