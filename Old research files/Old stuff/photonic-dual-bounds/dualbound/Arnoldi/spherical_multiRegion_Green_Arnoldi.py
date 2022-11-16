#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 11:13:15 2020

@author: pengning
"""

import numpy as np
import numpy.polynomial.polynomial as po
import matplotlib.pyplot as plt
from .spherical_Green_Taylor_Arnoldi_speedup import mp_re
from .shell_Green_Taylor_Arnoldi_spatialDiscretization import rgrid_Mmn_dot,rgrid_Nmn_dot, rgrid_Mmn_normsqr, rgrid_Nmn_normsqr, shell_Green_grid_Arnoldi_RgandImMmn_Uconverge, shell_Green_grid_Arnoldi_RgandImNmn_Uconverge
from .shell_Green_Taylor_Arnoldi_spatialDiscretization_mp import shell_Green_grid_Arnoldi_RgandImMmn_Uconverge_mp, shell_Green_grid_Arnoldi_RgandImNmn_Uconverge_mp
from .spherical_Green_Taylor_Arnoldi_speedup import rmnMnormsqr_Taylor, rmnNnormsqr_Taylor, speedup_Green_Taylor_Arnoldi_RgMmn_Uconverge, speedup_Green_Taylor_Arnoldi_RgNmn_Uconverge
import mpmath
from mpmath import mp

def mparr_to_npreal(mparr):
    arr = np.zeros(len(mparr))
    for i in range(len(mparr)):
        arr[i] = np.float(mp.nstr(mparr[i],17))
    return arr

def spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge(n,k,RPlist, invchi, gridpts=10000, mpdps=60, klim=25, Taylor_tol=1e-12, Unormtol=1e-8, veclim=3, delveclim=2, maxveclim=40):
    """
    generates a representation of the Green's function/Umatrix over spherical region of radius R
    with sub-bases with support in shell regions with boundaries delineated by RPlist
    this is done so that the projection operator for spatial projection based constraints is explicit
    the sub-regions are 0-RPlist[0], RPlist[0]-RPlist[1], ..., RPlist[-2]-RPlist[-1]
    RPlist[-1] is the radius of the entire bounding sphere
    the first region is an inner sphere, the other regions are cocentric shells
    note here we are still using the old convention for the U matrix to be consistent with older Arnoldi code
    in the new optimizations U = V^\dagger-1 - G^\dagger; here we calculate Uinv, and Uinv = V^-1-G
    """
    mp.dps = mpdps #set mpmath precision
    #first step: generate the sub-bases and sub-Gmat/Uinvs for each block
    regionnum = len(RPlist)
    unitRgdotRglist = np.zeros(regionnum, dtype=type(1j*mp.one)) #values needed for computing coupling between different sub-bases in Gmat
    unitRgdotOutlist = np.zeros(regionnum, dtype=type(1j*mp.one)) #stored using mpmath to avoid underflow when calculating Gmat couplings
    unitImdotOutlist = np.zeros(regionnum, dtype=type(1j*mp.one))
    
    subGmatlist = []
    vecnum = 0
    subbasis_head_indlist = []
    All_unitMvecs = []
    rgridlist = []
    for i in range(regionnum):
        print('M wave Region #', i)
        if i==0: #inner spherical region is special because it contains origin, use old mpmath Taylor Arnoldi code
            subbasis_head_indlist.append(0)
            rmnRgM, rnImM, unitrmnMpols, Uinv = speedup_Green_Taylor_Arnoldi_RgMmn_Uconverge(n,k,RPlist[0], klim=klim, Taylor_tol=Taylor_tol, invchi=invchi, Unormtol=Unormtol)
            unitRgdotRglist[0] = mp.sqrt(rmnMnormsqr_Taylor(n,k,RPlist[0],rmnRgM)) #unitRg dot Rg is just norm of the regular wave
            #for the inner sphere, the outgoing wave quantities are not relevant since the inner sphere contains origin
            subGmat = mp.eye(Uinv.rows)*invchi-Uinv
            subGmatlist.append(np.array(mpmath.fp.matrix(subGmat.tolist()).tolist()))
            vecnum += Uinv.rows
            
            #generate ptval representation for the Arnoldi basis to be outputted
            rgrid = np.linspace(0,RPlist[0],gridpts)
            rgridlist.append(rgrid)
            for i in range(len(unitrmnMpols)-1): #don't include the last unorthogonalized, unnormalized Arnoldi vector
                All_unitMvecs.append((k*rgrid)**n * po.polyval(k*rgrid, unitrmnMpols[i].coef))
        else:
            subbasis_head_indlist.append(vecnum)
            try:
                rgrid, rsqrgrid, rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Uinv, Gmat = shell_Green_grid_Arnoldi_RgandImMmn_Uconverge(n,k,RPlist[i-1],RPlist[i], invchi, gridpts=gridpts, Unormtol=Unormtol, maxveclim=maxveclim)
                OutMgrid = RgMgrid + 1j*ImMgrid
                unitRgdotRglist[i] = mp.sqrt(rgrid_Mmn_normsqr(RgMgrid,rsqrgrid,rdiffgrid))
                unitRgdotOutlist[i] = mp.mpc(rgrid_Mmn_dot(unitMvecs[0], OutMgrid, rsqrgrid,rdiffgrid))
                unitImdotOutlist[i] = mp.mpc(rgrid_Mmn_dot(unitMvecs[1], OutMgrid, rsqrgrid,rdiffgrid))
            except FloatingPointError:
                rgrid, rsqrgrid, rdiffgrid, RgMgrid, ImMgrid, unitMvecs, Uinv, Gmat = shell_Green_grid_Arnoldi_RgandImMmn_Uconverge_mp(n,k,RPlist[i-1],RPlist[i], invchi, gridpts=gridpts, Unormtol=Unormtol, maxveclim=maxveclim)
                OutMgrid = RgMgrid + 1j*ImMgrid
                unitRgdotRglist[i] = mp.sqrt(rgrid_Mmn_normsqr(RgMgrid,rsqrgrid,rdiffgrid))
                unitRgdotOutlist[i] = rgrid_Mmn_dot(unitMvecs[0], OutMgrid, rsqrgrid,rdiffgrid)
                unitImdotOutlist[i] = rgrid_Mmn_dot(unitMvecs[1], OutMgrid, rsqrgrid,rdiffgrid)
                Gmat = np.array(mpmath.fp.matrix(Gmat.tolist()).tolist())
            subGmatlist.append(Gmat)
            vecnum += Gmat.shape[0]
            All_unitMvecs.extend(unitMvecs[:-2]) #don't include the last two unorthogonalized, unnormalized Arnoldi vectors
            rgridlist.append(rgrid)
            
    subbasis_head_indlist.append(vecnum) #for bookkeeping convenience put the total number of basis vectors at end of the subbasis family head index list
    Gmat = np.zeros((vecnum,vecnum),dtype=np.complex) #the Green's function representation for the entire domain
    for i in range(regionnum):
        indstart = subbasis_head_indlist[i]; indend = subbasis_head_indlist[i+1]
        Gmat[indstart:indend,indstart:indend] = subGmatlist[i][:,:]

    #print('RgdotRgM', unitRgdotRglist)
    #print('RgdotOut', unitRgdotOutlist)
    #print('ImdotOut', unitImdotOutlist)
    
    #next generate the couplings between different subbases
    jkcubed = 1j * k**3
    for i in range(regionnum):
        Rgiind = subbasis_head_indlist[i]
        Imiind = Rgiind+1
        #first do regions lying within region #i
        for j in range(i):
            Rgjind = subbasis_head_indlist[j]
            Gmat[Rgjind,Rgiind] = np.complex(jkcubed * unitRgdotRglist[j] * unitRgdotOutlist[i])
            Gmat[Rgjind,Imiind] = np.complex(jkcubed * unitRgdotRglist[j] * unitImdotOutlist[i])
        #then do regions lying outside region #i
        for j in range(i+1,regionnum):
            Rgjind = subbasis_head_indlist[j]
            Imjind = Rgjind+1
            Gmat[Rgjind,Rgiind] = np.complex(jkcubed * unitRgdotOutlist[j] * unitRgdotRglist[i])
            Gmat[Imjind,Rgiind] = np.complex(jkcubed * unitImdotOutlist[j] * unitRgdotRglist[i])
    
    #prepare for output
    #outputting Rgnormlist is for use later to construct source vectors
    #outputting subbasis_head_indlist is for use later to construct projection matrices
    Uinv = invchi*np.eye(vecnum) - Gmat

    #create an rgrid over the entire domain and extend the ptval representation of all the subbases onto the entire domain, for potential plotting purposes later
    fullrgrid = rgridlist[0].copy()
    rboundaries = [0,gridpts]
    for i in range(1,len(rgridlist)):
        fullrgrid = np.concatenate((fullrgrid,rgridlist[i][1:])) #1: so we don't have overlapping grid points
        rboundaries.append(len(fullrgrid))
    
    All_fullr_unitMvecs = []
    for i in range(len(rgridlist)):
        for j in range(subbasis_head_indlist[i],subbasis_head_indlist[i+1]):
            vecgrid = np.zeros_like(fullrgrid)
            if i==0:
                #print(All_unitMvecs[j])
                vecgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitMvecs[j][:]))
            else:
                vecgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitMvecs[j][1:]))
            All_fullr_unitMvecs.append(vecgrid)

    return Gmat, Uinv, unitRgdotRglist, subbasis_head_indlist, fullrgrid, All_fullr_unitMvecs


def spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge(n,k,RPlist, invchi, gridpts=10000, mpdps=60, klim=25, Taylor_tol=1e-12, Unormtol=1e-8, veclim=3, delveclim=2, maxveclim=40):
    """
    generates a representation of the Green's function/Umatrix over spherical region of radius R
    with sub-bases with support in shell regions with boundaries delineated by RPlist
    this is done so that the projection operator for spatial projection based constraints is explicit
    the sub-regions are 0-RPlist[0], RPlist[0]-RPlist[1], ..., RPlist[-2]-RPlist[-1]
    RPlist[-1] is the radius of the entire bounding sphere
    the first region is an inner sphere, the other regions are cocentric shells
    note here we are still using the old convention for the U matrix to be consistent with older Arnoldi code
    in the new optimizations U = V^\dagger-1 - G^\dagger; here we calculate Uinv, and Uinv = V^-1-G
    """
    mp.dps = mpdps #set mpmath precision
    #first step: generate the sub-bases and sub-Gmat/Uinvs for each block
    regionnum = len(RPlist)
    unitRgdotRglist = np.zeros(regionnum, dtype=type(1j*mp.one)) #values needed for computing coupling between different sub-bases in Gmat
    unitRgdotOutlist = np.zeros(regionnum, dtype=type(1j*mp.one)) #stored using mpmath to avoid underflow when calculating Gmat couplings
    unitImdotOutlist = np.zeros(regionnum, dtype=type(1j*mp.one))
    
    subGmatlist = []
    vecnum = 0
    subbasis_head_indlist = []
    rgridlist = []
    All_unitBvecs = []; All_unitPvecs = []
    for i in range(regionnum):
        print('N wave Region #', i)
        if i==0: #inner spherical region is special because it contains origin, use old mpmath Taylor Arnoldi code
            subbasis_head_indlist.append(0)
            rmRgN_Bpol, rmRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols, unitrmnPpols, Uinv = speedup_Green_Taylor_Arnoldi_RgNmn_Uconverge(n,k,RPlist[0],klim=klim, Taylor_tol=Taylor_tol, invchi=invchi, Unormtol=Unormtol)
            unitRgdotRglist[0] = mp.sqrt(rmnNnormsqr_Taylor(n,k,RPlist[0],rmRgN_Bpol,rmRgN_Ppol)) #unitRg dot Rg is just norm of the regular wave
            #for the inner sphere, the outgoing wave quantities are not relevant since the inner sphere contains origin
            subGmat = mp.eye(Uinv.rows)*invchi-Uinv
            subGmatlist.append(np.array(mpmath.fp.matrix(subGmat.tolist()).tolist()))
            vecnum += Uinv.rows
            
            rgrid = np.linspace(0,RPlist[0],gridpts)
            rgridlist.append(rgrid)
            for i in range(len(unitrmnBpols)-1):
                All_unitBvecs.append((k*rgrid)**(n-1) * po.polyval(k*rgrid, unitrmnBpols[i].coef))
                All_unitPvecs.append((k*rgrid)**(n-1) * po.polyval(k*rgrid, unitrmnPpols[i].coef))
        else:
            subbasis_head_indlist.append(vecnum)
            try:
                rgrid, rsqrgrid, rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Uinv, Gmat = shell_Green_grid_Arnoldi_RgandImNmn_Uconverge(n,k,RPlist[i-1],RPlist[i],invchi,gridpts=gridpts, Unormtol=Unormtol, maxveclim=maxveclim)
                OutBgrid = RgBgrid + 1j*ImBgrid
                OutPgrid = RgPgrid + 1j*ImPgrid
                unitRgdotRglist[i] = mp.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid,rsqrgrid,rdiffgrid))
                unitRgdotOutlist[i] = mp.mpc(rgrid_Nmn_dot(unitBvecs[0],unitPvecs[0], OutBgrid,OutPgrid, rsqrgrid,rdiffgrid))
                unitImdotOutlist[i] = mp.mpc(rgrid_Nmn_dot(unitBvecs[1],unitPvecs[1], OutBgrid,OutPgrid, rsqrgrid,rdiffgrid))
            except FloatingPointError:
                rgrid, rsqrgrid, rdiffgrid, RgBgrid,RgPgrid, ImBgrid,ImPgrid, unitBvecs,unitPvecs, Uinv, Gmat = shell_Green_grid_Arnoldi_RgandImNmn_Uconverge_mp(n,k,RPlist[i-1],RPlist[i],invchi,gridpts=gridpts, Unormtol=Unormtol, maxveclim=maxveclim)
                OutBgrid = RgBgrid + 1j*ImBgrid
                OutPgrid = RgPgrid + 1j*ImPgrid
                unitRgdotRglist[i] = mp.sqrt(rgrid_Nmn_normsqr(RgBgrid,RgPgrid,rsqrgrid,rdiffgrid))
                unitRgdotOutlist[i] = mp.mpc(rgrid_Nmn_dot(unitBvecs[0],unitPvecs[0], OutBgrid,OutPgrid, rsqrgrid,rdiffgrid))
                unitImdotOutlist[i] = mp.mpc(rgrid_Nmn_dot(unitBvecs[1],unitPvecs[1], OutBgrid,OutPgrid, rsqrgrid,rdiffgrid))
                Gmat = np.array(mpmath.fp.matrix(Gmat.tolist()).tolist())
            subGmatlist.append(Gmat)
            vecnum += Gmat.shape[0]
            
            rgridlist.append(rgrid)
            All_unitBvecs.extend(unitBvecs[:-2])
            All_unitPvecs.extend(unitPvecs[:-2])
    
    subbasis_head_indlist.append(vecnum) #for bookkeeping convenience put the total number of basis vectors at end of the subbasis family head index list
    Gmat = np.zeros((vecnum,vecnum),dtype=np.complex) #the Green's function representation for the entire domain
    for i in range(regionnum):
        indstart = subbasis_head_indlist[i]; indend = subbasis_head_indlist[i+1]
        Gmat[indstart:indend,indstart:indend] = subGmatlist[i][:,:]

    #print('RgdotRgN', unitRgdotRglist)
    #print('RgdotOut', unitRgdotOutlist)
    #print('ImdotOut', unitImdotOutlist)
    
    #next generate the couplings between different subbases
    jkcubed = 1j * k**3
    for i in range(regionnum):
        Rgiind = subbasis_head_indlist[i]
        Imiind = Rgiind+1
        #first do regions lying within region #i
        for j in range(i):
            Rgjind = subbasis_head_indlist[j]
            Gmat[Rgjind,Rgiind] = np.complex(jkcubed * unitRgdotRglist[j] * unitRgdotOutlist[i])
            Gmat[Rgjind,Imiind] = np.complex(jkcubed * unitRgdotRglist[j] * unitImdotOutlist[i])
        #then do regions lying outside region #i
        for j in range(i+1,regionnum):
            Rgjind = subbasis_head_indlist[j]
            Imjind = Rgjind+1
            Gmat[Rgjind,Rgiind] = np.complex(jkcubed * unitRgdotOutlist[j] * unitRgdotRglist[i])
            Gmat[Imjind,Rgiind] = np.complex(jkcubed * unitImdotOutlist[j] * unitRgdotRglist[i])
    
    #prepare for output
    #outputting Rgnormlist is for use later to construct source vectors
    #outputting subbasis_head_indlist is for use later to construct projection matrices
    Uinv = invchi*np.eye(vecnum) - Gmat
    
    #create an rgrid over the entire domain and extend the ptval representation of all the subbases onto the entire domain, for potential plotting purposes later
    fullrgrid = rgridlist[0].copy()
    rboundaries = [0,gridpts]
    for i in range(1,len(rgridlist)):
        fullrgrid = np.concatenate((fullrgrid,rgridlist[i][1:])) #1: so we don't have overlapping grid points
        rboundaries.append(len(fullrgrid))
    
    All_fullr_unitBvecs = []; All_fullr_unitPvecs = []
    for i in range(len(rgridlist)):
        for j in range(subbasis_head_indlist[i],subbasis_head_indlist[i+1]):
            vecBgrid = np.zeros_like(fullrgrid)
            vecPgrid = np.zeros_like(fullrgrid)
            if i==0:
                #print(All_unitMvecs[j])
                vecBgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitBvecs[j][:]))
                vecPgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitPvecs[j][:]))
            else:
                vecBgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitBvecs[j][1:]))
                vecPgrid[rboundaries[i]:rboundaries[i+1]] = mparr_to_npreal(mp_re(All_unitPvecs[j][1:]))
            All_fullr_unitBvecs.append(vecBgrid)
            All_fullr_unitPvecs.append(vecPgrid)

    return Gmat, Uinv, unitRgdotRglist, subbasis_head_indlist, fullrgrid, All_fullr_unitBvecs,All_fullr_unitPvecs