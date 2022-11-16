#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  5 21:46:50 2020

@author: pengning
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
from .shell_domain import shell_rho_M, shell_rho_N
import mpmath
from mpmath import mp
from .dipole_field import mp_spherical_jn, mp_vec_spherical_jn, mp_spherical_yn, mp_vec_spherical_yn, mp_vec_spherical_djn, mp_vec_spherical_dyn
from .spherical_Green_Taylor_Arnoldi_speedup import mp_re, mp_im
from .shell_Green_Taylor_Arnoldi_spatialDiscretization import complex_to_mp, grid_integrate_trap, rgrid_Mmn_plot,rgrid_Nmn_plot, rgrid_Mmn_normsqr, rgrid_Nmn_normsqr, rgrid_Mmn_vdot,rgrid_Nmn_vdot


def mp_to_complex(mpcarr):
    a = np.zeros(len(mpcarr),dtype=np.complex)
    for i in range(len(mpcarr)):
        a[i] = np.complex(mpcarr[i])
    return a

def shell_Green_grid_Mmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, vecMgrid):
    """
    evaluates G(r,r')*vecM(r') over a shell region from R1 to R2
    the region coordinates are contained in rsqrgrid, a grid of r^2, and rdiffgrid, the distances between neighboring grid points; these instead of the original rgrid are given so that they only need to be computed once in main Arnoldi method
    """
    #rsqrgrid = rgrid**2
    #rdiffgrid = np.diff(rgrid)
    
    RgMvecMrsqr_grid = RgMgrid*vecMgrid*rsqrgrid
    Im_newvecMgrid = k**3 * grid_integrate_trap(RgMvecMrsqr_grid, rdiffgrid) * RgMgrid
    
    Re_ImMfactgrid = np.zeros_like(rsqrgrid, dtype=type(mp.one))
    Re_ImMfactgrid[1:] = k**3 * np.cumsum((RgMvecMrsqr_grid[:-1]+RgMvecMrsqr_grid[1:])*rdiffgrid/2.0)
    
    rev_ImMvecMrsqr_grid = np.flip(ImMgrid*vecMgrid*rsqrgrid) #reverse the grid direction to evaluate integrands of the form kr' to kR2
    
    Re_RgMfactgrid = np.zeros_like(rsqrgrid, dtype=type(mp.one))
    Re_RgMfactgrid[:-1] = k**3 * np.flip(np.cumsum( (rev_ImMvecMrsqr_grid[:-1]+rev_ImMvecMrsqr_grid[1:])*np.flip(rdiffgrid)/2.0 ))
    
    Re_newvecMgrid = -ImMgrid*Re_ImMfactgrid - RgMgrid*Re_RgMfactgrid
    
    return Re_newvecMgrid + 1j*Im_newvecMgrid


def shell_Green_grid_Arnoldi_RgandImMmn_step_mp(n,k, invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=False):
    """
    using a mpf valued grid
    this method does one more Arnoldi step, given existing Arnoldi vectors in unitMvecs
    the last two entries in unitMvecs is unitMvecs[-2]=G*unitMvecs[-4] and unitMvecs[-1]=G*unitMvecs[-3] without orthogonalization and normalization
    its indices -1 and -3 because we are alternatingly generating new vectors starting from either the RgM line or the ImM line
    so len(unitMvecs) = len(Gmat)+2 going in and going out of the method
    this is setup for most efficient iteration since G*unitMvec is only computed once
    the unitMvecs list is modified on spot; a new enlarged Gmat nparray is returned at the end
    for each iteration we only advance Gmat by 1 row and 1 column
    Gmat here is an mpmatrix
    """
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]

    vecnum = Gmat.rows
    for i in range(vecnum):
        coef = Gmat[i,vecnum-2]
        unitMvecs[-2] -= coef*unitMvecs[i]
        
    unitMvecs[-2][:] = mp_re(unitMvecs[-2][:]) #the Arnoldi vectors should all be real since RgM is a family head and only non-zero singular vector of AsymG
    
    norm = mp.sqrt(rgrid_Mmn_normsqr(unitMvecs[-2], rsqrgrid,rdiffgrid))
    unitMvecs[-2] /= norm
    
    if plotVectors:
        rgrid_Mmn_plot(unitMvecs[-2], rgrid)
    
    #get new vector
    newvecM = shell_Green_grid_Mmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, unitMvecs[-2])
    newvecM[:] = mp_re(newvecM)
    
    vecnum += 1
    Gmat.rows+=1; Gmat.cols+=1
    
    for i in range(vecnum):
        Gmat[i,vecnum-1] = rgrid_Mmn_vdot(unitMvecs[i], newvecM, rsqrgrid,rdiffgrid)
        Gmat[vecnum-1,i] = Gmat[i,vecnum-1]
    
    unitMvecs.append(newvecM) #append to end of unitMvecs for next round of iteration
    return Gmat
    
def shell_Green_grid_Arnoldi_RgandImMmn_Uconverge_mp(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, maxveclim=40, plotVectors=False):
    np.seterr(over='raise',under='raise',invalid='raise')
    #for high angular momentum number could have floating point issues; in this case, raise error. Outer method will catch the error and use the mpmath version instead
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    """
    RgMgrid = sp.spherical_jn(n, k*rgrid) #the argument for radial part of spherical waves is kr
    ImMgrid = sp.spherical_yn(n, k*rgrid)
    RgMgrid = RgMgrid.astype(np.complex)
    ImMgrid = ImMgrid.astype(np.complex)
    
    RgMgrid = complex_to_mp(RgMgrid)
    ImMgrid = complex_to_mp(ImMgrid)
    """
    RgMgrid = mp_vec_spherical_jn(n, k*rgrid)
    ImMgrid = mp_vec_spherical_yn(n, k*rgrid)
    
    vecRgMgrid = RgMgrid / mp.sqrt(rgrid_Mmn_normsqr(RgMgrid, rsqrgrid,rdiffgrid))
    
    vecImMgrid = ImMgrid - rgrid_Mmn_vdot(vecRgMgrid, ImMgrid, rsqrgrid,rdiffgrid)*vecRgMgrid
    vecImMgrid /= mp.sqrt(rgrid_Mmn_normsqr(vecImMgrid,rsqrgrid,rdiffgrid))
    
    if plotVectors:
        rgrid_Mmn_plot(vecRgMgrid,rgrid)
        rgrid_Mmn_plot(vecImMgrid,rgrid)
    
    unitMvecs = [vecRgMgrid,vecImMgrid]
    
    GvecRgMgrid = shell_Green_grid_Mmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, vecRgMgrid)
    GvecImMgrid = shell_Green_grid_Mmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgMgrid,ImMgrid, vecImMgrid)
    Gmat = mp.zeros(2,2)
    Gmat[0,0] = rgrid_Mmn_vdot(vecRgMgrid, GvecRgMgrid, rsqrgrid,rdiffgrid)
    Gmat[0,1] = rgrid_Mmn_vdot(vecRgMgrid, GvecImMgrid, rsqrgrid,rdiffgrid)
    Gmat[1,0] = Gmat[0,1]
    Gmat[1,1] = rgrid_Mmn_vdot(vecImMgrid,GvecImMgrid, rsqrgrid,rdiffgrid)
    Uinv = mp.eye(2)*invchi-Gmat

    unitMvecs.append(GvecRgMgrid)
    unitMvecs.append(GvecImMgrid) #append unorthogonalized, unnormalized Arnoldi vector for further iterations
    
    b = mp.matrix([mp.one])
    prevUnorm = 1 / Uinv[0,0]
    
    i=2
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_RgandImMmn_step_mp(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Gmat, plotVectors=plotVectors)
        i += 1
        print(i)
        if i==maxveclim:
            break
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = mp.eye(Gmat.rows)*invchi-Gmat
            b.rows = i
            x = mp.lu_solve(Uinv, b)
            Unorm = mp.norm(x)
            print('Unorm', Unorm, flush=True)
            if np.abs(prevUnorm-Unorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return rgrid,rsqrgrid,rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Uinv, Gmat


def shell_Green_grid_Nmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, vecBgrid,vecPgrid):
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
    
    Re_ImNfactgrid = np.zeros_like(rsqrgrid, dtype=type(1j*mp.one))
    Re_ImNfactgrid[1:] = k**3 * np.cumsum((RgNvecNrsqr_grid[:-1]+RgNvecNrsqr_grid[1:])*rdiffgrid/2.0)
    
    rev_ImNvecNrsqr_grid = np.flip((ImBgrid*vecBgrid + ImPgrid*vecPgrid) * rsqrgrid) #reverse the grid direction to evaluate integrands of the form kr' to kR2
    
    Re_RgNfactgrid = np.zeros_like(rsqrgrid, dtype=type(1j*mp.one))
    Re_RgNfactgrid[:-1] = k**3 * np.flip(np.cumsum( (rev_ImNvecNrsqr_grid[:-1]+rev_ImNvecNrsqr_grid[1:])*np.flip(rdiffgrid)/2.0 ))
    
    Re_newvecBgrid = -ImBgrid*Re_ImNfactgrid - RgBgrid*Re_RgNfactgrid
    Re_newvecPgrid = -ImPgrid*Re_ImNfactgrid - RgPgrid*Re_RgNfactgrid - vecPgrid #last term is delta contribution
    
    return Re_newvecBgrid + 1j*Im_newvecBgrid, Re_newvecPgrid + 1j*Im_newvecPgrid


def shell_Green_grid_Arnoldi_RgandImNmn_step_mp(n,k, invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=False):
    """
    this method does one more Arnoldi step, given existing Arnoldi vectors in unitNvecs
    the last two entries in unitMvecs is unitNvecs[-2]=G*unitNvecs[-4] and unitNvecs[-1]=G*unitNvecs[-3] without orthogonalization and normalization
    its indices -1 and -3 because we are alternatingly generating new vectors starting from either the RgN line or the ImN line
    so len(unitNvecs) = len(Gmat)+2 going in and going out of the method
    this is setup for most efficient iteration since G*unitNvec is only computed once
    the unitNvecs lists is modified on spot; a new enlarged Gmat mpmatrix is returned at the end
    for each iteration we only advance Gmat by 1 row and 1 column
    """
    #first, begin by orthogonalizing and normalizing unitMvecs[-1]

    vecnum = Gmat.rows
    for i in range(vecnum):
        coef = Gmat[i,vecnum-2]
        unitBvecs[-2] -= coef*unitBvecs[i]; unitPvecs[-2] -= coef*unitPvecs[i]
    #the Arnoldi vectors should all be real since RgM is a family head and only non-zero singular vector of AsymG
    unitBvecs[-2][:] = mp_re(unitBvecs[-2][:]); unitPvecs[-2][:] = mp_re(unitPvecs[-2][:])
    
    norm = mp.sqrt(rgrid_Nmn_normsqr(unitBvecs[-2],unitPvecs[-2], rsqrgrid,rdiffgrid))
    unitBvecs[-2] /= norm; unitPvecs[-2] /= norm
    
    if plotVectors:
        rgrid_Nmn_plot(unitBvecs[-2],unitPvecs[-2], rgrid)
    
    #get new vector
    newvecB,newvecP = shell_Green_grid_Nmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs[-2],unitPvecs[-2])
    newvecB[:] = mp_re(newvecB); newvecP[:] = mp_re(newvecP)
    
    vecnum += 1
    Gmat.rows+=1; Gmat.cols+=1
    for i in range(vecnum):
        Gmat[i,vecnum-1] = rgrid_Nmn_vdot(unitBvecs[i],unitPvecs[i], newvecB,newvecP, rsqrgrid,rdiffgrid)
        Gmat[vecnum-1,i] = Gmat[i,vecnum-1]
    
    unitBvecs.append(newvecB); unitPvecs.append(newvecP) #append to end of unitNvecs for next round of iteration
    return Gmat
    
def shell_Green_grid_Arnoldi_RgandImNmn_Uconverge_mp(n,k,R1,R2, invchi, gridpts=1000, Unormtol=1e-10, veclim=3, delveclim=2, maxveclim=40, plotVectors=False):
    np.seterr(over='raise',under='raise',invalid='raise')
    #for high angular momentum number could have floating point issues; in this case, raise error. Outer method will catch the error and use the mpmath version instead
    rgrid = np.linspace(R1,R2,gridpts)
    rsqrgrid = rgrid**2
    rdiffgrid = np.diff(rgrid)
    """
    RgBgrid = sp.spherical_jn(n, k*rgrid)/(k*rgrid) + sp.spherical_jn(n,k*rgrid,derivative=True) #the argument for radial part of spherical waves is kr
    RgPgrid = np.sqrt(n*(n+1))*sp.spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = sp.spherical_yn(n, k*rgrid)/(k*rgrid) + sp.spherical_yn(n,k*rgrid,derivative=True)
    ImPgrid = np.sqrt(n*(n+1))*sp.spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgBgrid = RgBgrid.astype(np.complex)
    RgPgrid = RgPgrid.astype(np.complex)
    ImBgrid = ImBgrid.astype(np.complex)
    ImPgrid = ImPgrid.astype(np.complex)
    
    RgBgrid = complex_to_mp(RgBgrid)
    RgPgrid = complex_to_mp(RgPgrid)
    ImBgrid = complex_to_mp(ImBgrid)
    ImPgrid = complex_to_mp(ImPgrid)
    """
    RgBgrid = mp_vec_spherical_jn(n,k*rgrid)/(k*rgrid) + mp_vec_spherical_djn(n,k*rgrid)
    RgPgrid = mp.sqrt(n*(n+1))*mp_vec_spherical_jn(n, k*rgrid)/(k*rgrid)
    ImBgrid = mp_vec_spherical_yn(n, k*rgrid)/(k*rgrid) + mp_vec_spherical_dyn(n,k*rgrid)
    ImPgrid = mp.sqrt(n*(n+1))*mp_vec_spherical_yn(n, k*rgrid)/(k*rgrid)
    
    RgN_normvec = mp.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid, rsqrgrid,rdiffgrid))
    RgN_vecBgrid = RgBgrid / RgN_normvec
    RgN_vecPgrid = RgPgrid / RgN_normvec
    
    #next generate the orthonormal head for the outgoing wave series
    coef = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, ImBgrid,ImPgrid, rsqrgrid,rdiffgrid)
    ImN_vecBgrid = ImBgrid - coef*RgN_vecBgrid
    ImN_vecPgrid = ImPgrid - coef*RgN_vecPgrid
    ImN_normvec = mp.sqrt(rgrid_Nmn_normsqr(ImN_vecBgrid,ImN_vecPgrid, rsqrgrid,rdiffgrid))
    ImN_vecBgrid /= ImN_normvec
    ImN_vecPgrid /= ImN_normvec
    
    if plotVectors:
        rgrid_Nmn_plot(mp_to_complex(RgN_vecBgrid),mp_to_complex(RgN_vecPgrid),rgrid)
        rgrid_Nmn_plot(mp_to_complex(ImN_vecBgrid),mp_to_complex(ImN_vecPgrid),rgrid)
        
    unitBvecs = [RgN_vecBgrid,ImN_vecBgrid]
    unitPvecs = [RgN_vecPgrid,ImN_vecPgrid]
    
    
    GvecRgBgrid, GvecRgPgrid = shell_Green_grid_Nmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, RgN_vecBgrid,RgN_vecPgrid)
    
    GvecImBgrid, GvecImPgrid = shell_Green_grid_Nmn_vec_mp(n,k, rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, ImN_vecBgrid,ImN_vecPgrid)
    Gmat = mp.zeros(2,2)
    Gmat[0,0] = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, GvecRgBgrid,GvecRgPgrid, rsqrgrid,rdiffgrid)
    Gmat[0,1] = rgrid_Nmn_vdot(RgN_vecBgrid,RgN_vecPgrid, GvecImBgrid,GvecImPgrid, rsqrgrid,rdiffgrid)
    Gmat[1,0] = Gmat[0,1]
    Gmat[1,1] = rgrid_Nmn_vdot(ImN_vecBgrid,ImN_vecPgrid, GvecImBgrid,GvecImPgrid, rsqrgrid,rdiffgrid)
    Uinv = mp.eye(2)*invchi-Gmat

    unitBvecs.append(GvecRgBgrid); unitPvecs.append(GvecRgPgrid)
    unitBvecs.append(GvecImBgrid); unitPvecs.append(GvecImPgrid) #append unorthogonalized, unnormalized Arnoldi vector for further iterations
    
    b = mp.matrix([mp.one])
    prevUnorm = 1 / Uinv[0,0]
    
    i=2
    while i<veclim:
        Gmat = shell_Green_grid_Arnoldi_RgandImNmn_step_mp(n,k,invchi, rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Gmat, plotVectors=plotVectors)
        i += 1
        print(i)
        if i==maxveclim:
            break
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            Uinv = mp.eye(Gmat.rows)*invchi-Gmat
            b.rows = i
            x = mp.lu_solve(Uinv, b)
            Unorm = mp.norm(x)
            print('Unorm', Unorm, flush=True)
            if np.abs(prevUnorm-Unorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
    
    return rgrid,rsqrgrid,rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Uinv, Gmat
