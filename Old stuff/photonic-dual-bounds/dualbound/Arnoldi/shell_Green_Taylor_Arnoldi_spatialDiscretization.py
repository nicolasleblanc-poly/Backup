#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 21:10:21 2020

@author: pengning

does the Green's function Arnoldi iteration over a shell domain for spherical waves
nice analytical properties of polynomial representation lost when using shell domain leaving out origin
try going back to spatial discretization idea instead
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
from .shell_domain import shell_rho_M, shell_rho_N
import mpmath
from mpmath import mp

def grid_integrate_trap(integrandgrid,diffgrid):
    #integrate a spatial grid representation of the integrand using trapezoid rule
    return np.sum((integrandgrid[:-1]+integrandgrid[1:])*diffgrid/2.0)

def rgrid_Mmn_normsqr(vecMgrid, rsqrgrid, rdiffgrid):
    return np.real(grid_integrate_trap(np.conj(vecMgrid)*vecMgrid*rsqrgrid, rdiffgrid))

def rgrid_Mmn_dot(vecM1grid, vecM2grid, rsqrgrid, rdiffgrid):
    return grid_integrate_trap(vecM1grid*vecM2grid*rsqrgrid, rdiffgrid)

def rgrid_Mmn_vdot(vecM1grid, vecM2grid, rsqrgrid, rdiffgrid):
    return grid_integrate_trap(np.conj(vecM1grid)*vecM2grid*rsqrgrid, rdiffgrid)


def rgrid_Mmn_plot(vecMgrid, rgrid):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    ax1.plot(rgrid,np.real(vecMgrid))
    ax2.plot(rgrid,np.imag(vecMgrid))
    plt.show()


def shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, vecMgrid):
    """
    evaluates G(r,r')*vecM(r') over a shell region from R1 to R2
    the region coordinates are contained in rsqrgrid, a grid of r^2, and rdiffgrid, the distances between neighboring grid points; these instead of the original rgrid are given so that they only need to be computed once in main Arnoldi method
    """
    #rsqrgrid = rgrid**2
    #rdiffgrid = np.diff(rgrid)
    
    RgMvecMrsqr_grid = RgMgrid*vecMgrid*rsqrgrid
    Im_newvecMgrid = k**3 * grid_integrate_trap(RgMvecMrsqr_grid, rdiffgrid) * RgMgrid
    
    Re_ImMfactgrid = np.zeros_like(rsqrgrid, dtype=np.complex)
    Re_ImMfactgrid[1:] = k**3 * np.cumsum((RgMvecMrsqr_grid[:-1]+RgMvecMrsqr_grid[1:])*rdiffgrid/2.0)
    
    rev_ImMvecMrsqr_grid = np.flip(ImMgrid*vecMgrid*rsqrgrid) #reverse the grid direction to evaluate integrands of the form kr' to kR2
    
    Re_RgMfactgrid = np.zeros_like(rsqrgrid, dtype=np.complex)
    Re_RgMfactgrid[:-1] = k**3 * np.flip(np.cumsum( (rev_ImMvecMrsqr_grid[:-1]+rev_ImMvecMrsqr_grid[1:])*np.flip(rdiffgrid)/2.0 ))
    
    Re_newvecMgrid = -ImMgrid*Re_ImMfactgrid - RgMgrid*Re_RgMfactgrid
    
    return Re_newvecMgrid + 1j*Im_newvecMgrid


def shell_Green_grid_Arnoldi_Mmn_oneshot(n,k,R1,R2, invchi, vecnum, gridpts=200):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgMgrid = sp.spherical_jn(n, k*rgrid) #the argument for radial part of spherical waves is kr
    ImMgrid = sp.spherical_yn(n, k*rgrid)
    RgMgrid = RgMgrid.astype(np.complex)
    ImMgrid = ImMgrid.astype(np.complex)
    
    vecMgrid = RgMgrid / np.sqrt(rgrid_Mmn_normsqr(RgMgrid, rsqrgrid,rdiffgrid))
    rgrid_Mmn_plot(vecMgrid, rgrid)
    unitMvecs = [vecMgrid]
    
    for i in range(1,vecnum):
        newvecMgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs[-1])
        newvecMgrid[:] = np.real(newvecMgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Mmn_plot(newvecMgrid, rgrid)
        for j in range(len(unitMvecs)):
            unitMvec = unitMvecs[j]
            coeff = rgrid_Mmn_vdot(unitMvec, newvecMgrid, rsqrgrid,rdiffgrid)
            newvecMgrid -= coeff*unitMvec
        
        newvecMgrid /= np.sqrt(rgrid_Mmn_normsqr(newvecMgrid, rsqrgrid,rdiffgrid))
        rgrid_Mmn_plot(newvecMgrid, rgrid)
        print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitMvecs.append(newvecMgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GMjgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs[j])
            Green[i,j] = rgrid_Mmn_vdot(unitMvecs[i],GMjgrid, rsqrgrid,rdiffgrid)
    
    print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    return Green, Umat


def shell_Green_grid_Arnoldi_Mmn_step(n,k, invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=False):
    """
    this method does one more Arnoldi step, given existing Arnoldi vectors in unitMvecs
    the last entry in unitMvecs is G*unitMvecs[-2] without orthogonalization and normalization
    so len(unitMvecs) = len(Gmat)+1 going in and going out of the method
    this is setup for most efficient iteration since G*unitMvec is only computed once
    the unitMvecs list is modified on spot; a new enlarged Gmat nparray is returned at the end
    """
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]
    #use relation U = V^{-1} - G
    """
    see comment for analogous method for N waves, shell_Green_grid_Arnoldi_Nmn_step
    coef1 = Gmat[-1,-1]
    unitMvecs[-1] -= coef1*unitMvecs[-2]
    
    if Gmat.shape[0]>1: #since G has symmetric Arnoldi representation (so tridiagonal), G*M_j has non-zero overlap with M_j and M_{j-1}
        coef2 = Gmat[-2,-1]
        unitMvecs[-1] -= coef2*unitMvecs[-3]
    
    unitMvecs[-1][:] = np.real(unitMvecs[-1][:])
    """
    vecnum = Gmat.shape[0]
    for i in range(vecnum):
        coef = rgrid_Mmn_vdot(unitMvecs[i], unitMvecs[-1], rsqrgrid,rdiffgrid)
        unitMvecs[-1] -= coef*unitMvecs[i]
    unitMvecs[-1][:] = np.real(unitMvecs[-1][:])
    
    norm = np.sqrt(rgrid_Mmn_normsqr(unitMvecs[-1], rsqrgrid,rdiffgrid))
    unitMvecs[-1] /= norm
    
    if plotVectors:
        rgrid_Mmn_plot(unitMvecs[-1], rgrid)
    
    #get new vector
    newvecM = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, unitMvecs[-1])
    
    newvecM[:] = np.real(newvecM)
    
    newGmat = np.zeros((Gmat.shape[0]+1,Gmat.shape[1]+1), dtype=np.complex)
    newGmat[:-1,:-1] = Gmat[:,:]
    
    newGmat[-1,-1] = rgrid_Mmn_vdot(unitMvecs[-1], newvecM, rsqrgrid,rdiffgrid)
    newGmat[-2,-1] = rgrid_Mmn_vdot(unitMvecs[-2], newvecM, rsqrgrid,rdiffgrid)
    newGmat[-1,-2] = newGmat[-2,-1]
    
    unitMvecs.append(newvecM) #append to end of unitMvecs for next round of iteration
    return newGmat
    
def shell_Green_grid_Arnoldi_Mmn_Uconverge(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, plotVectors=False):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgMgrid = sp.spherical_jn(n, k*rgrid) #the argument for radial part of spherical waves is kr
    ImMgrid = sp.spherical_yn(n, k*rgrid)
    RgMgrid = RgMgrid.astype(np.complex)
    ImMgrid = ImMgrid.astype(np.complex)
    
    vecMgrid = RgMgrid / np.sqrt(rgrid_Mmn_normsqr(RgMgrid, rsqrgrid,rdiffgrid))
    unitMvecs = [vecMgrid]
    
    if plotVectors:
        rgrid_Mmn_plot(vecMgrid, rgrid)
    
    GvecMgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, vecMgrid)
    Gmat = np.array([[rgrid_Mmn_vdot(vecMgrid, GvecMgrid, rsqrgrid,rdiffgrid)]], dtype=np.complex)
    Uinv = invchi*np.eye(1)-Gmat
    unitMvecs.append(GvecMgrid) #append unorthogonalized, unnormalized Arnoldi vector for further iterations
    
    prevUnorm = 1.0/Uinv[0,0]
    
    i=1
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_Mmn_step(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=plotVectors)
        i += 1
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = invchi*np.eye(Gmat.shape[0])-Gmat
            b = np.zeros((Uinv.shape[0],1))
            b[0] = 1.0
            x = np.linalg.solve(Uinv,b)
            Unorm = np.linalg.norm(x)
            print('Unorm:', Unorm)
            if np.abs(Unorm-prevUnorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return RgMgrid, ImMgrid, unitMvecs, Uinv, Gmat
                                          


                
def rgrid_Nmn_dot(vecB1grid,vecP1grid, vecB2grid,vecP2grid, rsqrgrid,rdiffgrid):
    return grid_integrate_trap((vecB1grid*vecB2grid+vecP1grid*vecP2grid)*rsqrgrid, rdiffgrid)

def rgrid_Nmn_vdot(vecB1grid,vecP1grid, vecB2grid,vecP2grid, rsqrgrid,rdiffgrid):
    return grid_integrate_trap((np.conj(vecB1grid)*vecB2grid+np.conj(vecP1grid)*vecP2grid)*rsqrgrid, rdiffgrid)

def rgrid_Nmn_normsqr(vecBgrid,vecPgrid, rsqrgrid,rdiffgrid):
    return np.real(rgrid_Nmn_vdot(vecBgrid,vecPgrid, vecBgrid,vecPgrid, rsqrgrid,rdiffgrid))

def rgrid_Nmn_plot(vecBgrid,vecPgrid, rgrid):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(nrows=1, ncols=4,figsize=(10,4))
    ax1.plot(rgrid,np.real(vecBgrid))
    ax2.plot(rgrid,np.real(vecPgrid))
    ax3.plot(rgrid,np.imag(vecBgrid))
    ax4.plot(rgrid,np.imag(vecPgrid))
    ax1.set_title('B real'); ax2.set_title('P real'); ax3.set_title('B imag'); ax4.set_title('P imag')
    plt.show()
    

def shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, vecBgrid,vecPgrid):
    """
    evaluates G(r,r')*vecN(r') over a shell region from R1 to R2
    the region coordinates are contained in rsqrgrid, a grid of r^2, and rdiffgrid, the distances between neighboring grid points; these instead of the original rgrid are given so that they only need to be computed once in main Arnoldi method
    """
    #rsqrgrid = rgrid**2
    #rdiffgrid = np.diff(rgrid)
    
    RgNvecNrsqr_grid = (RgBgrid*vecBgrid+RgPgrid*vecPgrid)*rsqrgrid
    imfac = k**3 * grid_integrate_trap(RgNvecNrsqr_grid, rdiffgrid)
    Im_newvecBgrid = imfac * RgBgrid
    Im_newvecPgrid = imfac * RgPgrid
    
    Re_ImNfactgrid = np.zeros_like(rsqrgrid, dtype=np.complex)
    Re_ImNfactgrid[1:] = k**3 * np.cumsum((RgNvecNrsqr_grid[:-1]+RgNvecNrsqr_grid[1:])*rdiffgrid/2.0)
    
    rev_ImNvecNrsqr_grid = np.flip((ImBgrid*vecBgrid + ImPgrid*vecPgrid) * rsqrgrid) #reverse the grid direction to evaluate integrands of the form kr' to kR2
    
    Re_RgNfactgrid = np.zeros_like(rsqrgrid, dtype=np.complex)
    Re_RgNfactgrid[:-1] = k**3 * np.flip(np.cumsum( (rev_ImNvecNrsqr_grid[:-1]+rev_ImNvecNrsqr_grid[1:])*np.flip(rdiffgrid)/2.0 ))
    
    Re_newvecBgrid = -ImBgrid*Re_ImNfactgrid - RgBgrid*Re_RgNfactgrid
    Re_newvecPgrid = -ImPgrid*Re_ImNfactgrid - RgPgrid*Re_RgNfactgrid - vecPgrid #last term is delta contribution
    
    return Re_newvecBgrid + 1j*Im_newvecBgrid, Re_newvecPgrid + 1j*Im_newvecPgrid

def shell_Green_grid_Arnoldi_Nmn_oneshot(n,k,R1,R2, invchi, vecnum, gridpts=200):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    vecBgrid = RgBgrid / normvec
    vecPgrid = RgPgrid / normvec

    rgrid_Nmn_plot(vecBgrid,vecPgrid, rgrid)
    unitBvecs = [vecBgrid]; unitPvecs = [vecPgrid]
    
    for i in range(1,vecnum):
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-1],unitPvecs[-1])
        newvecBgrid[:] = np.real(newvecBgrid)
        newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = np.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    #print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    
    Umatw,Umatv = np.linalg.eig(Umat)
    print(Umatw)
    print('v0', Umatv[:,0])
    for i in range(len(Umatw)):
        #if np.abs(Umatw[i]-1-invchi)<1e-2*np.abs(1+invchi):
        if np.abs(np.imag(Umatw[i])-np.imag(invchi))<1e-4*np.abs(np.imag(invchi)):
            print(Umatw[i])
            print('v', Umatv[:,i])
            testvecB = np.zeros_like(unitBvecs[0],dtype=np.complex)
            testvecP = np.zeros_like(unitPvecs[0],dtype=np.complex)
            for j in range(vecnum):
                testvecB += Umatv[j,i]*unitBvecs[j]
                testvecP += Umatv[j,i]*unitPvecs[j]
            rgrid_Nmn_plot(testvecB,testvecP,rgrid)
            rgrid_Nmn_plot(ImBgrid,ImPgrid,rgrid)
            print(rgrid_Nmn_vdot(testvecB,testvecP,ImBgrid,ImPgrid,rsqrgrid,rdiffgrid))
            
    return Green, Umat


def shell_Green_grid_Arnoldi_Nmn_step(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=False):
    """
    this method does one more Arnoldi step, given existing N-type Arnoldi vectors stored in (unitBvecs, unitPvecs)
    the last entry in unitB/Pvecs is G*unitNvecs[-2] without orthogonalization and normalization
    so len(unitBvecs) = len(Gmat)+1 going in and going out of the method
    this is setup for most efficient iteration since G*unitNvec is only computed once
    the unitNvecs lists is modified on spot; a new enlarged Gmat nparray is returned at the end
    """
    
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]
    #use relation U = V^{-1} - G
    """
    it seems that when using grid based discretization, discretization error pushes the Arnoldi process away
    from true tridiagonality; there is small non-zero values in Gmat off the tri-diagonal.
    We take a middle ground: ignore the non-tridiagonal parts of Gmat due to discretization error,
    but when orthogonalizing the Arnoldi vectors apply all previous vectors instead of just the closest two,
    to maintain orthogonality up to eps for the Arnoldi vectors in the grid representation
    coef1 = Gmat[-1,-1]
    unitBvecs[-1] -= coef1*unitBvecs[-2]; unitPvecs[-1] -= coef1*unitPvecs[-2]
    
    if Gmat.shape[0]>1: #since G has symmetric Arnoldi representation (so tridiagonal), G*N_j has non-zero overlap with N_j and N_{j-1}
        coef2 = Gmat[-2,-1]
        unitBvecs[-1] -= coef2*unitBvecs[-3]; unitPvecs[-1] -= coef2*unitPvecs[-3]
    
    unitBvecs[-1][:] = np.real(unitBvecs[-1][:]); unitPvecs[-1][:] = np.real(unitPvecs[-1][:])
    """
    vecnum = Gmat.shape[0]
    for i in range(vecnum):
        coef = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], unitBvecs[-1],unitPvecs[-1], rsqrgrid,rdiffgrid)
        unitBvecs[-1] -= coef*unitBvecs[i]; unitPvecs[-1] -= coef*unitPvecs[i]
    unitBvecs[-1][:] = np.real(unitBvecs[-1][:]); unitPvecs[-1][:] = np.real(unitPvecs[-1][:])
    
    norm = np.sqrt(rgrid_Nmn_normsqr(unitBvecs[-1],unitPvecs[-1], rsqrgrid,rdiffgrid))
    unitBvecs[-1] /= norm; unitPvecs[-1] /= norm
    
    if plotVectors:
        rgrid_Nmn_plot(unitBvecs[-1],unitPvecs[-1], rgrid)
    
    #get new vector
    newvecB,newvecP = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-1],unitPvecs[-1])
    newvecB[:] = np.real(newvecB); newvecP[:] = np.real(newvecP)
    
    newGmat = np.zeros((Gmat.shape[0]+1,Gmat.shape[1]+1), dtype=np.complex)
    newGmat[:-1,:-1] = Gmat[:,:]
    
    newGmat[-1,-1] = rgrid_Nmn_vdot(unitBvecs[-1],unitPvecs[-1], newvecB,newvecP, rsqrgrid,rdiffgrid)
    newGmat[-2,-1] = rgrid_Nmn_vdot(unitBvecs[-2],unitPvecs[-2], newvecB,newvecP, rsqrgrid,rdiffgrid)
    newGmat[-1,-2] = newGmat[-2,-1]
    
    unitBvecs.append(newvecB); unitPvecs.append(newvecP) #append to end of unitB/Pvecs for next round of iteration
    return newGmat

def shell_Green_grid_Arnoldi_Nmn_Uconverge(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, plotVectors=False):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    vecBgrid = RgBgrid / normvec
    vecPgrid = RgPgrid / normvec

    unitBvecs = [vecBgrid]; unitPvecs = [vecPgrid]
    
    if plotVectors:
        rgrid_Nmn_plot(vecBgrid,vecPgrid, rgrid)
    
    GvecBgrid, GvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, vecBgrid,vecPgrid)
    Gmat = np.array([[rgrid_Nmn_vdot(vecBgrid,vecPgrid, GvecBgrid,GvecPgrid, rsqrgrid,rdiffgrid)]])
    Uinv = invchi*np.eye(1)-Gmat
    unitBvecs.append(GvecBgrid); unitPvecs.append(GvecPgrid)
    
    prevUnorm = 1.0/Uinv[0,0]
    
    i=1
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_Nmn_step(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=plotVectors)
        i += 1
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = invchi*np.eye(Gmat.shape[0])-Gmat
            b = np.zeros((Uinv.shape[0],1))
            b[0] = 1.0
            x = np.linalg.solve(Uinv,b)
            Unorm = np.linalg.norm(x)
            print('Unorm:', Unorm)
            if np.abs(Unorm-prevUnorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    
    Proj = np.zeros((veclim,veclim),dtype=np.complex)
    for i in range(veclim):
        for j in range(veclim):
            Proj[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], unitBvecs[j],unitPvecs[j], rsqrgrid,rdiffgrid)
    
    return RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Uinv, Gmat, Proj



"""
Projection operators for rgrid shell domains
"""



"""
rgrid Arnoldi iteration starting with both regular and outgoing wave, for use in multiple region Arnoldi
"""

def shell_Green_grid_Arnoldi_RgandImMmn_oneshot(n,k,R1,R2, invchi, vecnum, gridpts=1000):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgMgrid = sp.spherical_jn(n, k*rgrid) #the argument for radial part of spherical waves is kr
    ImMgrid = sp.spherical_yn(n, k*rgrid)
    RgMgrid = RgMgrid.astype(np.complex)
    ImMgrid = ImMgrid.astype(np.complex)
    
    vecRgMgrid = RgMgrid / np.sqrt(rgrid_Mmn_normsqr(RgMgrid, rsqrgrid,rdiffgrid))
    
    vecImMgrid = ImMgrid - rgrid_Mmn_vdot(vecRgMgrid, ImMgrid, rsqrgrid,rdiffgrid)*vecRgMgrid
    vecImMgrid /= np.sqrt(rgrid_Mmn_normsqr(vecImMgrid,rsqrgrid,rdiffgrid))
    
    rgrid_Mmn_plot(vecRgMgrid,rgrid)
    rgrid_Mmn_plot(vecImMgrid,rgrid)
    unitMvecs = [vecRgMgrid,vecImMgrid]
    
    for i in range(2,vecnum):
        #alternate between RgM generated waves and ImM generated waves
        newvecMgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs[-2])
        #newvecMgrid[:] = np.real(newvecMgrid)
        print('before orthogonalization and normalization:')
        rgrid_Mmn_plot(newvecMgrid, rgrid)
        for j in range(len(unitMvecs)):
            unitMvec = unitMvecs[j]
            coeff = rgrid_Mmn_vdot(unitMvec, newvecMgrid, rsqrgrid,rdiffgrid)
            newvecMgrid -= coeff*unitMvec
        
        newvecMgrid /= np.sqrt(rgrid_Mmn_normsqr(newvecMgrid, rsqrgrid,rdiffgrid))
        rgrid_Mmn_plot(newvecMgrid, rgrid)
        print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitMvecs.append(newvecMgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GMjgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs[j])
            Green[i,j] = rgrid_Mmn_vdot(unitMvecs[i],GMjgrid, rsqrgrid,rdiffgrid)
    
    Umat = invchi*np.eye(vecnum)-Green
    print(Green)
    return Green, Umat


def shell_Green_grid_Arnoldi_RgandImMmn_step(n,k, invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=False):
    """
    this method does one more Arnoldi step, given existing Arnoldi vectors in unitMvecs
    the last two entries in unitMvecs is unitMvecs[-2]=G*unitMvecs[-4] and unitMvecs[-1]=G*unitMvecs[-3] without orthogonalization and normalization
    its indices -1 and -3 because we are alternatingly generating new vectors starting from either the RgM line or the ImM line
    so len(unitMvecs) = len(Gmat)+2 going in and going out of the method
    this is setup for most efficient iteration since G*unitMvec is only computed once
    the unitMvecs list is modified on spot; a new enlarged Gmat nparray is returned at the end
    for each iteration we only advance Gmat by 1 row and 1 column
    """
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]

    vecnum = Gmat.shape[0]
    for i in range(vecnum):
        coef = Gmat[i,-2]
        unitMvecs[-2] -= coef*unitMvecs[i]
        
    unitMvecs[-2][:] = np.real(unitMvecs[-2][:]) #the Arnoldi vectors should all be real since RgM is a family head and only non-zero singular vector of AsymG
    
    norm = np.sqrt(rgrid_Mmn_normsqr(unitMvecs[-2], rsqrgrid,rdiffgrid))
    unitMvecs[-2] /= norm
    
    if plotVectors:
        rgrid_Mmn_plot(unitMvecs[-2], rgrid)
    
    #get new vector
    newvecM = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, unitMvecs[-2])
    newvecM[:] = np.real(newvecM)
    
    vecnum += 1
    newGmat = np.zeros((Gmat.shape[0]+1,Gmat.shape[1]+1), dtype=np.complex)
    newGmat[:-1,:-1] = Gmat[:,:]
    for i in range(vecnum):
        newGmat[i,-1] = rgrid_Mmn_vdot(unitMvecs[i], newvecM, rsqrgrid,rdiffgrid)
        newGmat[-1,i] = newGmat[i,-1]
    
    unitMvecs.append(newvecM) #append to end of unitMvecs for next round of iteration
    return newGmat
    
def shell_Green_grid_Arnoldi_RgandImMmn_Uconverge(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, maxveclim=40, plotVectors=False):
    np.seterr(over='raise',under='raise',invalid='raise')
    #for high angular momentum number could have floating point issues; in this case, raise error. Outer method will catch the error and use the mpmath version instead
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgMgrid = sp.spherical_jn(n, k*rgrid) #the argument for radial part of spherical waves is kr
    ImMgrid = sp.spherical_yn(n, k*rgrid)
    RgMgrid = RgMgrid.astype(np.complex)
    ImMgrid = ImMgrid.astype(np.complex)
    
    
    vecRgMgrid = RgMgrid / np.sqrt(rgrid_Mmn_normsqr(RgMgrid, rsqrgrid,rdiffgrid))
    
    vecImMgrid = ImMgrid - rgrid_Mmn_vdot(vecRgMgrid, ImMgrid, rsqrgrid,rdiffgrid)*vecRgMgrid
    vecImMgrid /= np.sqrt(rgrid_Mmn_normsqr(vecImMgrid,rsqrgrid,rdiffgrid))
    
    if plotVectors:
        rgrid_Mmn_plot(vecRgMgrid,rgrid)
        rgrid_Mmn_plot(vecImMgrid,rgrid)
    
    unitMvecs = [vecRgMgrid,vecImMgrid]
    
    GvecRgMgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, vecRgMgrid)
    GvecImMgrid = shell_Green_grid_Mmn_vec(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, vecImMgrid)
    Gmat = np.zeros((2,2), dtype=np.complex)
    Gmat[0,0] = rgrid_Mmn_vdot(vecRgMgrid, GvecRgMgrid, rsqrgrid,rdiffgrid)
    Gmat[0,1] = rgrid_Mmn_vdot(vecRgMgrid, GvecImMgrid, rsqrgrid,rdiffgrid)
    Gmat[1,0] = Gmat[0,1]
    Gmat[1,1] = rgrid_Mmn_vdot(vecImMgrid,GvecImMgrid, rsqrgrid,rdiffgrid)
    Uinv = invchi*np.eye(2)-Gmat

    unitMvecs.append(GvecRgMgrid)
    unitMvecs.append(GvecImMgrid) #append unorthogonalized, unnormalized Arnoldi vector for further iterations
    
    prevUnorm = 1.0/Uinv[0,0]
    
    i=2
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_RgandImMmn_step(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=plotVectors)
        i += 1
        if i==maxveclim:
            break
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = invchi*np.eye(Gmat.shape[0])-Gmat
            b = np.zeros((Uinv.shape[0],1))
            b[0] = 1.0
            x = np.linalg.solve(Uinv,b)
            Unorm = np.linalg.norm(x)
            print('Unorm:', Unorm, flush=True)
            if np.abs(Unorm-prevUnorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Uinv, Gmat





def shell_Green_grid_Arnoldi_RgandImNmn_oneshot(n,k,R1,R2, invchi, vecnum, gridpts=1000):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    RgN_normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    RgN_vecBgrid = RgBgrid / RgN_normvec
    RgN_vecPgrid = RgPgrid / RgN_normvec
    
    #next generate the orthonormal head for the outgoing wave series
    coef = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, ImBgrid,ImPgrid, rsqrgrid,rdiffgrid)
    ImN_vecBgrid = ImBgrid - coef*RgN_vecBgrid
    ImN_vecPgrid = ImPgrid - coef*RgN_vecPgrid
    ImN_normvec = np.sqrt(rgrid_Nmn_normsqr(ImN_vecBgrid,ImN_vecPgrid, rsqrgrid,rdiffgrid))
    ImN_vecBgrid /= ImN_normvec
    ImN_vecPgrid /= ImN_normvec
    
    rgrid_Nmn_plot(RgN_vecBgrid,RgN_vecPgrid,rgrid)
    rgrid_Nmn_plot(ImN_vecBgrid,ImN_vecPgrid,rgrid)
    unitBvecs = [RgN_vecBgrid,ImN_vecBgrid]
    unitPvecs = [RgN_vecPgrid,ImN_vecPgrid]
    
    for i in range(2,vecnum):
        #alternate between RgN generated waves and ImN generated waves
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-2],unitPvecs[-2])
        #newvecBgrid[:] = np.real(newvecBgrid)
        #newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = np.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    return Green, Umat

def shell_Green_grid_Arnoldi_RgandImNmn_step(n,k, invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=False):
    """
    this method does one more Arnoldi step, given existing Arnoldi vectors in unitNvecs
    the last two entries in unitMvecs is unitNvecs[-2]=G*unitNvecs[-4] and unitNvecs[-1]=G*unitNvecs[-3] without orthogonalization and normalization
    its indices -1 and -3 because we are alternatingly generating new vectors starting from either the RgN line or the ImN line
    so len(unitNvecs) = len(Gmat)+2 going in and going out of the method
    this is setup for most efficient iteration since G*unitNvec is only computed once
    the unitNvecs lists is modified on spot; a new enlarged Gmat nparray is returned at the end
    for each iteration we only advance Gmat by 1 row and 1 column
    """
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]

    vecnum = Gmat.shape[0]
    for i in range(vecnum):
        coef = Gmat[i,-2]
        unitBvecs[-2] -= coef*unitBvecs[i]; unitPvecs[-2] -= coef*unitPvecs[i]
    #the Arnoldi vectors should all be real since RgM is a family head and only non-zero singular vector of AsymG
    unitBvecs[-2][:] = np.real(unitBvecs[-2][:]); unitPvecs[-2][:] = np.real(unitPvecs[-2][:])
    
    norm = np.sqrt(rgrid_Nmn_normsqr(unitBvecs[-2],unitPvecs[-2], rsqrgrid,rdiffgrid))
    unitBvecs[-2] /= norm; unitPvecs[-2] /= norm
    
    if plotVectors:
        rgrid_Nmn_plot(unitBvecs[-2],unitPvecs[-2], rgrid)
    
    #get new vector
    newvecB,newvecP = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-2],unitPvecs[-2])
    newvecB[:] = np.real(newvecB); newvecP[:] = np.real(newvecP)
    
    vecnum += 1
    newGmat = np.zeros((Gmat.shape[0]+1,Gmat.shape[1]+1), dtype=np.complex)
    newGmat[:-1,:-1] = Gmat[:,:]
    for i in range(vecnum):
        newGmat[i,-1] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], newvecB,newvecP, rsqrgrid,rdiffgrid)
        newGmat[-1,i] = newGmat[i,-1]
    
    unitBvecs.append(newvecB); unitPvecs.append(newvecP) #append to end of unitNvecs for next round of iteration
    return newGmat
    
def shell_Green_grid_Arnoldi_RgandImNmn_Uconverge(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, maxveclim=40, plotVectors=False):
    np.seterr(over='raise',under='raise',invalid='raise')
    #for high angular momentum number could have floating point issues; in this case, raise error. Outer method will catch the error and use the mpmath version instead
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    RgN_normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    RgN_vecBgrid = RgBgrid / RgN_normvec
    RgN_vecPgrid = RgPgrid / RgN_normvec
    
    #next generate the orthonormal head for the outgoing wave series
    coef = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, ImBgrid,ImPgrid, rsqrgrid,rdiffgrid)
    ImN_vecBgrid = ImBgrid - coef*RgN_vecBgrid
    ImN_vecPgrid = ImPgrid - coef*RgN_vecPgrid
    ImN_normvec = np.sqrt(rgrid_Nmn_normsqr(ImN_vecBgrid,ImN_vecPgrid, rsqrgrid,rdiffgrid))
    ImN_vecBgrid /= ImN_normvec
    ImN_vecPgrid /= ImN_normvec
    
    if plotVectors:
        rgrid_Nmn_plot(RgN_vecBgrid,RgN_vecPgrid,rgrid)
        rgrid_Nmn_plot(ImN_vecBgrid,ImN_vecPgrid,rgrid)
        
    unitBvecs = [RgN_vecBgrid,ImN_vecBgrid]
    unitPvecs = [RgN_vecPgrid,ImN_vecPgrid]
    
    
    GvecRgBgrid, GvecRgPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, RgN_vecBgrid,RgN_vecPgrid)
    
    GvecImBgrid, GvecImPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, ImN_vecBgrid,ImN_vecPgrid)
    Gmat = np.zeros((2,2), dtype=np.complex)
    Gmat[0,0] = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, GvecRgBgrid,GvecRgPgrid, rsqrgrid,rdiffgrid)
    Gmat[0,1] = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, GvecImBgrid,GvecImPgrid, rsqrgrid,rdiffgrid)
    Gmat[1,0] = Gmat[0,1]
    Gmat[1,1] = rgrid_Nmn_vdot(ImN_vecBgrid,ImN_vecPgrid, GvecImBgrid,GvecImPgrid, rsqrgrid,rdiffgrid)
    Uinv = invchi*np.eye(2)-Gmat

    unitBvecs.append(GvecRgBgrid); unitPvecs.append(GvecRgPgrid)
    unitBvecs.append(GvecImBgrid); unitPvecs.append(GvecImPgrid) #append unorthogonalized, unnormalized Arnoldi vector for further iterations
    
    prevUnorm = 1.0/Uinv[0,0]
    
    i=2
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_RgandImNmn_step(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=plotVectors)
        i += 1
        if i==maxveclim:
            break
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = invchi*np.eye(Gmat.shape[0])-Gmat
            b = np.zeros((Uinv.shape[0],1))
            b[0] = 1.0
            x = np.linalg.solve(Uinv,b)
            Unorm = np.linalg.norm(x)
            print('Unorm:', Unorm, flush=True)
            if np.abs(Unorm-prevUnorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return rgrid, rsqrgrid, rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Uinv, Gmat

    

def complex_to_mp(nparr):
    mparr = []
    for i in range(len(nparr)):
        mparr.append(mp.mpc(nparr[i]))
    return np.array(mparr)

def mp_to_complex(mpcplx):
    mpreal = mp.re(mpcplx); mpimag = mp.im(mpcplx)
    flreal = np.float(mp.nstr(mpreal, mp.dps))
    flimag = np.float(mp.nstr(mpimag, mp.dps))
    return flreal + 1j*flimag

def shell_grid_Green_Nmn_vec_mptest(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, vecBgrid,vecPgrid):
    """
    test to see if slight asymmetry of resulting Green matrix is due to numerical inaccuracy or an implementation bug
    conclusion is that it is discretization error
    evaluates G(r,r')*vecN(r') over a shell region from R1 to R2
    the region coordinates are contained in rsqrgrid, a grid of r^2, and rdiffgrid, the distances between neighboring grid points; these instead of the original rgrid are given so that they only need to be computed once in main Arnoldi method
    """
    #rsqrgrid = rgrid**2
    #rdiffgrid = np.diff(rgrid)
    
    RgNvecNrsqr_grid = (RgBgrid*vecBgrid+RgPgrid*vecPgrid)*rsqrgrid
    imfac = k**3 * grid_integrate_trap(RgNvecNrsqr_grid, rdiffgrid)
    Im_newvecBgrid = imfac * RgBgrid
    Im_newvecPgrid = imfac * RgPgrid
    
    Re_ImNfactgrid = np.zeros_like(rsqrgrid, dtype=type(1j*mp.one))
    Re_ImNfactgrid[1:] = k**3 * np.cumsum((RgNvecNrsqr_grid[:-1]+RgNvecNrsqr_grid[1:])*rdiffgrid/2.0)
    
    rev_ImNvecNrsqr_grid = np.flip((ImBgrid*vecBgrid + ImPgrid*vecPgrid) * rsqrgrid) #reverse the grid direction to evaluate integrands of the form kr' to kR2
    
    Re_RgNfactgrid = np.zeros_like(rsqrgrid, dtype=type(1j*mp.one))
    Re_RgNfactgrid[:-1] = k**3 * np.flip(np.cumsum( (rev_ImNvecNrsqr_grid[:-1]+rev_ImNvecNrsqr_grid[1:])*np.flip(rdiffgrid)/2.0 ))
    
    Re_newvecBgrid = -ImBgrid*Re_ImNfactgrid - RgBgrid*Re_RgNfactgrid
    Re_newvecPgrid = -ImPgrid*Re_ImNfactgrid - RgPgrid*Re_RgNfactgrid - vecPgrid #last term is delta contribution
    
    return Re_newvecBgrid + 1j*Im_newvecBgrid, Re_newvecPgrid + 1j*Im_newvecPgrid

def shell_Green_Taylor_Arnoldi_Nmn_oneshot_mptest(n,k,R1,R2, invchi, vecnum, gridpts=200):
    rgrid = np.linspace(R1,R2,gridpts)
    
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = complex_to_mp(RgBgrid)
    RgPgrid = complex_to_mp(RgPgrid)
    ImBgrid = complex_to_mp(ImBgrid)
    ImPgrid = complex_to_mp(ImPgrid)
    
    rgrid = complex_to_mp(rgrid)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    normvec = mp.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    vecBgrid = RgBgrid / normvec
    vecPgrid = RgPgrid / normvec

    #rgrid_Nmn_plot(vecBgrid.astype(np.complex), vecPgrid.astype(np.complex), rgrid)
    unitBvecs = [vecBgrid]; unitPvecs = [vecPgrid]
    
    for i in range(1,vecnum):
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-1],unitPvecs[-1])
        #newvecBgrid[:] = np.real(newvecBgrid)
        #newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        #rgrid_Nmn_plot(newvecBgrid.astype(np.complex), newvecPgrid.astype(np.complex), rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = mp.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        #rgrid_Nmn_plot(newvecBgrid.astype(np.complex), newvecPgrid.astype(np.complex), rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=type(1j*mp.one))
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    #print(Green)
    Uinv = np.eye(vecnum)*invchi - Green
    return Green, Uinv


from scipy.interpolate import interp1d
from scipy.integrate import quad
def shell_Green_grid_Arnoldi_Nmn_check_darkeig(n,k,R1,R2, invchi, vecnum, gridpts=200):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    vecBgrid = RgBgrid / normvec
    vecPgrid = RgPgrid / normvec

    rgrid_Nmn_plot(vecBgrid,vecPgrid, rgrid)
    unitBvecs = [vecBgrid]; unitPvecs = [vecPgrid]
    
    for i in range(1,vecnum):
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-1],unitPvecs[-1])
        newvecBgrid[:] = np.real(newvecBgrid)
        newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = np.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    #print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    
    Umatw,Umatv = np.linalg.eig(Umat)
    print(Umatw)
    print('v0', Umatv[:,0])
    for i in range(len(Umatw)):
        #if np.abs(Umatw[i]-1-invchi)<1e-2*np.abs(1+invchi):
        if np.abs(np.imag(Umatw[i])-np.imag(invchi))<1e-4*np.abs(np.imag(invchi)):
            print(Umatw[i])
            print('v', Umatv[:,i])
            testvecB = np.zeros_like(unitBvecs[0],dtype=np.complex)
            testvecP = np.zeros_like(unitPvecs[0],dtype=np.complex)
            for j in range(vecnum):
                testvecB += Umatv[j,i]*unitBvecs[j]
                testvecP += Umatv[j,i]*unitPvecs[j]
            rgrid_Nmn_plot(testvecB,testvecP,rgrid)
            rgrid_Nmn_plot(ImBgrid,ImPgrid,rgrid)
            print(rgrid_Nmn_vdot(testvecB,testvecP,ImBgrid,ImPgrid,rsqrgrid,rdiffgrid))
            
            testBfunc = interp1d(rgrid,np.real(testvecB),kind='cubic')
            testPfunc = interp1d(rgrid,np.real(testvecP),kind='cubic')
            
            
            highresrgrid = np.linspace(R1,R2,gridpts*200)
            highresRgBgrid = sp.spherical_jn(n, k*highresrgrid)/(k*highresrgrid) + sp.spherical_jn(n,k*highresrgrid,derivative=True) #the argument for radial part of spherical waves is kr
            highresRgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*highresrgrid)/(k*highresrgrid)
            highresImBgrid = sp.spherical_yn(n, k*highresrgrid)/(k*highresrgrid) + sp.spherical_yn(n,k*highresrgrid,derivative=True)
            highresImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*highresrgrid)/(k*highresrgrid)
            
            highres_testBgrid = testBfunc(highresrgrid)
            highres_testPgrid = testPfunc(highresrgrid)
            highresrsqrgrid = highresrgrid**2
            highresrdiffgrid = np.diff(highresrgrid)
            highres_imageBgrid, highres_imagePgrid = shell_Green_grid_Nmn_vec(1,2*np.pi, highresrsqrgrid, highresrdiffgrid, highresRgBgrid,highresRgPgrid, highresImBgrid,highresImPgrid, highres_testBgrid,highres_testPgrid)
            rgrid_Nmn_plot(highres_imageBgrid,highres_imagePgrid, highresrgrid)
            
            RgBfunc = interp1d(highresrgrid, highresRgBgrid, kind='cubic')
            RgPfunc = interp1d(highresrgrid, highresRgPgrid, kind='cubic')
            ImBfunc = interp1d(highresrgrid, highresImBgrid, kind='cubic')
            ImPfunc = interp1d(highresrgrid, highresImPgrid, kind='cubic')
            
            Rgorthog = lambda r: (testBfunc(r)*RgBfunc(r)+testPfunc(r)*RgPfunc(r)) * r**2
            Imorthog = lambda r: (testBfunc(r)*ImBfunc(r)+testPfunc(r)*ImPfunc(r)) * r**2
            print('Rg orthogonality:', quad(Rgorthog, R1,R2))
            print('Im orthogonality:', quad(Imorthog, R1,R2))
    
        
    
    return Green, Umat


def shell_Green_grid_Arnoldi_RgandImNmn_check_darkeig(n,k,R1,R2, invchi, vecnum, gridpts=1000):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    RgN_normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    RgN_vecBgrid = RgBgrid / RgN_normvec
    RgN_vecPgrid = RgPgrid / RgN_normvec
    
    #next generate the orthonormal head for the outgoing wave series
    coef = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, ImBgrid,ImPgrid, rsqrgrid,rdiffgrid)
    ImN_vecBgrid = ImBgrid - coef*RgN_vecBgrid
    ImN_vecPgrid = ImPgrid - coef*RgN_vecPgrid
    ImN_normvec = np.sqrt(rgrid_Nmn_normsqr(ImN_vecBgrid,ImN_vecPgrid, rsqrgrid,rdiffgrid))
    ImN_vecBgrid /= ImN_normvec
    ImN_vecPgrid /= ImN_normvec
    
    rgrid_Nmn_plot(RgN_vecBgrid,RgN_vecPgrid,rgrid)
    rgrid_Nmn_plot(ImN_vecBgrid,ImN_vecPgrid,rgrid)
    unitBvecs = [RgN_vecBgrid,ImN_vecBgrid]
    unitPvecs = [RgN_vecPgrid,ImN_vecPgrid]
    
    for i in range(2,vecnum):
        #alternate between RgN generated waves and ImN generated waves
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-2],unitPvecs[-2])
        newvecBgrid[:] = np.real(newvecBgrid)
        newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = np.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    
    Umatw,Umatv = np.linalg.eig(Umat)
    print(Umatw)
    print('v0', Umatv[:,0])
    for i in range(len(Umatw)):
        if np.abs(Umatw[i]-1-invchi)<1e-2*np.abs(1+invchi):
        #if np.abs(np.imag(Umatw[i])-np.imag(invchi))<1e-4*np.abs(np.imag(invchi)):
            print(Umatw[i])
            print('v', Umatv[:,i])
            testvecB = np.zeros_like(unitBvecs[0],dtype=np.complex)
            testvecP = np.zeros_like(unitPvecs[0],dtype=np.complex)
            for j in range(vecnum):
                testvecB += Umatv[j,i]*unitBvecs[j]
                testvecP += Umatv[j,i]*unitPvecs[j]
            rgrid_Nmn_plot(testvecB,testvecP,rgrid)
            rgrid_Nmn_plot(ImBgrid,ImPgrid,rgrid)
            print(rgrid_Nmn_vdot(testvecB,testvecP,ImBgrid,ImPgrid,rsqrgrid,rdiffgrid))
            
            testBfunc = interp1d(rgrid,np.real(testvecB),kind='cubic')
            testPfunc = interp1d(rgrid,np.real(testvecP),kind='cubic')
            
            highresrgrid = np.linspace(R1,R2,gridpts*20)
            highresRgBgrid = sp.spherical_jn(n, k*highresrgrid)/(k*highresrgrid) + sp.spherical_jn(n,k*highresrgrid,derivative=True) #the argument for radial part of spherical waves is kr
            highresRgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*highresrgrid)/(k*highresrgrid)
            highresImBgrid = sp.spherical_yn(n, k*highresrgrid)/(k*highresrgrid) + sp.spherical_yn(n,k*highresrgrid,derivative=True)
            highresImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*highresrgrid)/(k*highresrgrid)
            
            RgBfunc = interp1d(highresrgrid, highresRgBgrid, kind='cubic')
            RgPfunc = interp1d(highresrgrid, highresRgPgrid, kind='cubic')
            ImBfunc = interp1d(highresrgrid, highresImBgrid, kind='cubic')
            ImPfunc = interp1d(highresrgrid, highresImPgrid, kind='cubic')
            
            Rgorthog = lambda r: (testBfunc(r)*RgBfunc(r)+testPfunc(r)*RgPfunc(r)) * r**2
            Imorthog = lambda r: (testBfunc(r)*ImBfunc(r)+testPfunc(r)*ImPfunc(r)) * r**2
            print('Rg orthogonality:', quad(Rgorthog, R1,R2))
            print('Im orthogonality:', quad(Imorthog, R1,R2))
    
    return Green, Umat
    


######Arnoldi processes with random initial vectors
def shell_Green_grid_Arnoldi_Nmn_randinit(n,k,R1,R2, invchi, vecnum, gridpts=200):
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    normvec = np.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    vecBgrid = RgBgrid / normvec
    vecPgrid = RgPgrid / normvec

    vecBgrid = np.random.uniform()*np.sin(k*rgrid*np.random.uniform()) + np.random.uniform()*np.cos(k*rgrid*np.random.uniform())
    vecPgrid = np.random.uniform()*np.sin(k*rgrid*np.random.uniform()) + np.random.uniform()*np.cos(k*rgrid*np.random.uniform())
    normvec = np.sqrt(rgrid_Nmn_normsqr(vecBgrid,vecPgrid, rsqrgrid,rdiffgrid))
    vecBgrid /= normvec
    vecPgrid /= normvec

    rgrid_Nmn_plot(vecBgrid,vecPgrid, rgrid)
    unitBvecs = [vecBgrid]; unitPvecs = [vecPgrid]
    
    for i in range(1,vecnum):
        newvecBgrid, newvecPgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-1],unitPvecs[-1])
        #newvecBgrid[:] = np.real(newvecBgrid)
        #newvecPgrid[:] = np.real(newvecPgrid)
        
        print('before orthogonalization and normalization:')
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        for j in range(len(unitBvecs)):
            unitBvec = unitBvecs[j]; unitPvec = unitPvecs[j]
            coeff = rgrid_Nmn_vdot(unitBvec,unitPvec, newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid)
            newvecBgrid -= coeff*unitBvec; newvecPgrid -= coeff*unitPvec
        
        normvec = np.sqrt(rgrid_Nmn_normsqr(newvecBgrid,newvecPgrid, rsqrgrid,rdiffgrid))
        newvecBgrid /= normvec; newvecPgrid /= normvec
        
        rgrid_Nmn_plot(newvecBgrid,newvecPgrid, rgrid)
        #print(rgrid_Mmn_vdot(RgMgrid, newvecMgrid, rsqrgrid,rdiffgrid))
        unitBvecs.append(newvecBgrid); unitPvecs.append(newvecPgrid)
    
    Green = np.zeros((vecnum,vecnum), dtype=np.complex)
    
    for i in range(vecnum):
        for j in range(vecnum):
            GNj_Bgrid, GNj_Pgrid = shell_Green_grid_Nmn_vec(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[j],unitPvecs[j])
            Green[i,j] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], GNj_Bgrid,GNj_Pgrid, rsqrgrid,rdiffgrid)
    
    #print(Green)
    Umat = np.eye(vecnum)*invchi - Green
    
    Umatw,Umatv = np.linalg.eig(Umat)
    print(Umatw)
    #print('v0', Umatv[:,0])
    for i in range(len(Umatw)):
        testvecB = np.zeros_like(unitBvecs[0],dtype=np.complex)
        testvecP = np.zeros_like(unitPvecs[0],dtype=np.complex)
        for j in range(vecnum):
            testvecB += Umatv[j,i]*unitBvecs[j]
            testvecP += Umatv[j,i]*unitPvecs[j]
        
        RgN_ortho = rgrid_Nmn_vdot(testvecB,testvecP, RgBgrid,RgPgrid, rsqrgrid,rdiffgrid)
        ImN_ortho = rgrid_Nmn_vdot(testvecB,testvecP, ImBgrid,ImPgrid, rsqrgrid,rdiffgrid)
        
        if np.abs(RgN_ortho)<1e-4 and np.abs(ImN_ortho)<1e-4:
            print(Umatw[i])
            print('v', Umatv[:,i])
            print('RgN dot product', RgN_ortho)
            print('ImN dot product', ImN_ortho)
            rgrid_Nmn_plot(testvecB,testvecP,rgrid)

    return Green, Umat