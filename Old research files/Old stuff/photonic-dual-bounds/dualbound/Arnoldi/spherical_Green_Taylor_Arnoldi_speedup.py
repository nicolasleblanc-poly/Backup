#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 11:16:24 2019

@author: pengning
"""

import time
import numpy as np
import numpy.polynomial.polynomial as po
import scipy.special as sp
import mpmath
from mpmath import mp
import matplotlib.pyplot as plt
from .dipole_field import mp_spherical_jn, mp_spherical_djn, mp_spherical_yn, mp_spherical_dyn, mp_vec_spherical_jn, mp_vec_spherical_djn, mp_vec_spherical_yn, mp_vec_spherical_dyn
from .spherical_domain import mp_rho_M, mp_rho_N, rho_M, rho_N

axisfont = {'fontsize':'14'}

def mp_re(mpcarr):
    rearr = mp.zero*np.zeros_like(mpcarr)
    for i in range(len(mpcarr)):
        rearr[i] = mp.re(mpcarr[i])
    return rearr

def mp_im(mpcarr):
    imarr = mp.zero*np.zeros_like(mpcarr)
    for i in range(len(mpcarr)):
        imarr[i] = mp.im(mpcarr[i])
    return imarr

def mp_to_complex(mpcplx):
    mpreal = mp.re(mpcplx); mpimag = mp.im(mpcplx)
    flreal = np.float(mp.nstr(mpreal, mp.dps))
    flimag = np.float(mp.nstr(mpimag, mp.dps))
    return flreal + 1j*flimag

def plot_Taylor(powers,coeffs,lb,rb,pointnum=100,lstyle='-'):
    grid = np.linspace(lb,rb,pointnum)
    mpgrid = mp.one*grid
    series = mp.zero*np.zeros_like(grid,dtype=np.float)
    for i in range(len(powers)):
        series += coeffs[i] * mpgrid**powers[i]
    plt.plot(grid,series,linestyle=lstyle)
    
def plot_rmnNpol(n,rmnBpolcoef,rmnPpolcoef,lb,rb,pointnum=200):
    grid = np.linspace(np.float(lb),np.float(rb),pointnum)
    mpgrid = mp.one*grid
    fig,axs = plt.subplots(nrows=1, ncols=2)
#    print(po.polyval(grid,Bpolcoef))
    axs[0].plot(grid, mpgrid**(n-1)*po.polyval(grid,rmnBpolcoef))
    axs[1].plot(grid, mpgrid**(n-1)*po.polyval(grid,rmnPpolcoef))
    plt.show()
    
def plot_rmnMpol(n, rmnMpolcoef, lb,rb,pointnum=100):
    grid = np.linspace(np.float(lb),np.float(rb),pointnum)
    mpgrid = mp.one*grid
    plt.figure()
    plt.plot(grid, mpgrid**n * po.polyval(grid,rmnMpolcoef))
    plt.show()
    
    
def plot_compare_jn_yn_series(n,kR,klim=10):
    pow_jn, coe_jn, pow_yn, coe_yn = get_Taylor_jn_yn(n,kR,klim)
    pow_jn = np.array(pow_jn); pow_yn = np.array(pow_yn)
    coe_jn = np.array(coe_jn); coe_yn = np.array(coe_yn)

    krlist = mp.one*np.linspace(0.02*kR,kR,100)
    jn_sum = mp.zero*np.ones_like(krlist)
    yn_sum = mp.zero*np.ones_like(krlist)
    for i in range(len(pow_jn)):
        jn_sum += coe_jn[i] * krlist**pow_jn[i]
        yn_sum += coe_yn[i] * krlist**pow_yn[i]
        
    plt.figure()
    plt.plot(krlist,mp_vec_spherical_jn(n,krlist))
    plt.plot(krlist,jn_sum,'--')
    plt.show()
    plt.figure()
    plt.plot(krlist,mp_vec_spherical_yn(n,krlist))
    plt.plot(krlist,yn_sum,'--')
    plt.show()

def plot_compare_hn_dhn(n,kR,klim=10,tol=1e-4):
    pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_hndiv, coe_hndiv, pow_dhn, coe_dhn = get_Taylor_jndiv_djn_hndiv_dhn(n,kR,klim,tol)
    
    krlist = mp.one*np.linspace(0.1*kR,kR,100)
    
    plt.figure()
    plt.plot(krlist, mp_vec_spherical_jn(n, krlist)/krlist)
    plot_Taylor(pow_hndiv,mp_re(coe_hndiv), 0.1*kR,kR, lstyle='--')
    plt.show()

    plt.figure()
    plt.plot(krlist, mp_vec_spherical_djn(n, krlist))
    plot_Taylor(pow_dhn,mp_re(coe_dhn), 0.1*kR,kR, lstyle='--')
    plt.show()
    
    plt.figure()
    plt.plot(krlist, mp_vec_spherical_yn(n, krlist)/krlist)
    plot_Taylor(pow_hndiv,mp_im(coe_hndiv), 0.1*kR,kR, lstyle='--')
    plt.show()

    plt.figure()
    plt.plot(krlist, mp_vec_spherical_dyn(n, krlist))
    plot_Taylor(pow_dhn,mp_im(coe_dhn), 0.1*kR,kR, lstyle='--')
    plt.show()
    
    plt.figure()
    plt.plot(krlist, mp_vec_spherical_dyn(n,krlist) + (mp_vec_spherical_yn(n,krlist)/krlist))
    plt.show()
    return mp_vec_spherical_dyn(1, 0.05) + mp_vec_spherical_yn(1, 0.05)/0.05
    
def get_Taylor_jn_yn(n,kR,klim=5,tol=1e-6):
    #returns a Taylor series of jn that is accurate at jn(kR) up to abstol
    jn_kR = mp_spherical_jn(n,kR) #value for comparison
    yn_kR = mp_spherical_yn(n,kR)
    series_jnkR = mp.zero #current (kR) value of truncated series
    series_ynkR = mp.zero
    
    pow_jn = []; pow_yn = [] #the powers represented in the series
    coe_jn = []; coe_yn = [] #the coeffs of the represented powers
    
    k=0 #summation variable
    #klim = 5; delta_klim = 3;
    fack = mp.one #k! at any given step
    dfac2n2k1 = mp.fac2(2*n+2*k+1) #(2n+2k+1)!! at any given step
    dfac2nm2km1 = mp.fac2(2*n-2*k-1)
    twopowk = mp.one #2^k at any given step
    
    del_klim=3
    while k<=klim:
        pow_jn.append(n+2*k)
        pow_yn.append(2*k-n-1)
        neg = (-1)**(k % 2)
        coe_jn.append(neg / (twopowk*fack*dfac2n2k1))
        if k<=n:
            coe_yn.append(-dfac2nm2km1/(twopowk*fack))
        else:
            coe_yn.append((-1)**((n+k+1)%2) / (twopowk*fack*dfac2km2nm1))
        
        series_jnkR += coe_jn[-1] * kR**pow_jn[-1]
        series_ynkR += coe_yn[-1] * kR**pow_yn[-1]
        #print(jn_kR,series_jnkR)
        if k==klim and (np.abs(series_jnkR-jn_kR)>tol*np.abs(jn_kR) or np.abs(series_ynkR-yn_kR)>tol*np.abs(yn_kR)):
            klim += del_klim
        
        k += 1
        fack *= k
        dfac2n2k1 *= 2*n+2*k+1
        twopowk *= 2
        if k<=n:
            dfac2nm2km1 /= 2*n-2*(k-1)-1
        elif k==n+1:
            dfac2km2nm1 = 1
        else:
            dfac2km2nm1 *= 2*k-2*n-1
    
    return pow_jn, coe_jn, pow_yn, coe_yn
    
def get_Taylor_jndiv_djn_yndiv_dyn(n,kR, klim=10,tol=None):
    #make kR MPMATH OBJECT
    kR = mp.mpf(kR)
    del_klim = 3
    jndiv_kR = mp_spherical_jn(n,kR)/kR; djn_kR = mp_spherical_djn(n,kR)
    yndiv_kR = mp_spherical_yn(n,kR)/kR; dyn_kR = mp_spherical_dyn(n,kR)
    series_jndiv=mp.zero; series_djn=mp.zero
    series_yndiv=mp.zero; series_dyn=mp.zero
    pow_jndiv = []; pow_djn = []
    pow_yndiv = []; pow_dyn = []
    coe_jndiv = []; coe_djn = []
    coe_yndiv = []; coe_dyn = []
    k=0 #summation variable
    twopowk = mp.one #2^k
    fack = mp.one #k!
    dfac2n2k1 = mp.fac2(2*n+2*k+1)
    dfac2nm2km1 = mp.fac2(2*n-2*k-1)
    while k<=klim:
        pow_jndiv.append(n+2*k-1)
        coe_jndiv.append((-1)**(k%2) / (twopowk*fack*dfac2n2k1))
        pow_djn.append(pow_jndiv[-1])
        coe_djn.append((n+2*k)*coe_jndiv[-1])
        
        pow_yndiv.append(2*k-n-2)
        pow_dyn.append(pow_yndiv[-1])
        if k<=n:
            coe_yndiv.append(-dfac2nm2km1 / (twopowk*fack))
        else:
            coe_yndiv.append((-1)**((n+k+1)%2) / (twopowk*fack*dfac2km2nm1))
        coe_dyn.append((2*k-n-1)*coe_yndiv[-1])
        
        if tol!=None: #if we set a tolerance, extend klim if tolerance has not been met
            series_jndiv += coe_jndiv[-1]*kR**pow_jndiv[-1]
            series_djn += coe_djn[-1]*kR**pow_djn[-1]
            series_yndiv += coe_yndiv[-1]*kR**pow_yndiv[-1]
            series_dyn += coe_dyn[-1]*kR**pow_dyn[-1]
            
            if k==klim:
                #print(k)
                #print(series_jndiv-jndiv_kR)
                #print(series_djn-djn_kR)
                #print(series_yndiv-yndiv_kR)
                #print(series_dyn-dyn_kR)
                if np.abs(series_jndiv-jndiv_kR)>max(tol,tol*np.abs(jndiv_kR)):
                    klim += del_klim
                elif np.abs(series_djn-djn_kR)>max(tol,tol*np.abs(djn_kR)):
                    klim += del_klim
                elif np.abs(series_yndiv-yndiv_kR)>max(tol,tol*np.abs(yndiv_kR)):
                    klim += del_klim
                elif np.abs(series_dyn-dyn_kR)>max(tol,tol*np.abs(dyn_kR)):
                    klim += del_klim
        
        k += 1
        twopowk *= 2
        fack *= k
        dfac2n2k1 *= 2*n+2*k+1
        if k<=n:
            dfac2nm2km1 /= 2*n-2*(k-1)-1
        elif k==n+1:
            dfac2km2nm1 = 1
        else:
            dfac2km2nm1 *= 2*k-2*n-1
    
    return pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_yndiv, coe_yndiv, pow_dyn, coe_dyn

def get_Taylor_jndiv_djn_hndiv_dhn(n,kR, klim=10,tol=None):
    pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_yndiv, coe_yndiv, pow_dyn, coe_dyn = get_Taylor_jndiv_djn_yndiv_dyn(n,kR,klim,tol)
    
    pow_hndiv = []; coe_hndiv = []
    pow_dhn = []; coe_dhn = []
    
    #do merge sort to combine the jn and yn power series to get the hn series
    pj = 0; py = 0
    while pj<len(pow_jndiv) or py<len(pow_yndiv):
        if py==len(pow_yndiv) or pow_jndiv[pj]<pow_yndiv[py]:
            pow_hndiv.append(pow_jndiv[pj])
            coe_hndiv.append(coe_jndiv[pj])
            pj += 1
        elif pj==len(pow_jndiv) or pow_yndiv[py]<pow_jndiv[pj]:
            pow_hndiv.append(pow_yndiv[py])
            coe_hndiv.append(1j*coe_yndiv[py])
            py += 1
        else:
            pow_hndiv.append(pow_jndiv[pj])
            coe_hndiv.append(coe_jndiv[pj] + 1j*coe_yndiv[py])
            pj += 1; py += 1
            
    pj = 0; py = 0
    while pj<len(pow_djn) or py<len(pow_dyn):
        if py==len(pow_dyn) or pow_djn[pj]<pow_dyn[py]:
            pow_dhn.append(pow_djn[pj])
            coe_dhn.append(coe_djn[pj])
            pj += 1
        elif pj==len(pow_djn) or pow_dyn[py]<pow_djn[pj]:
            pow_dhn.append(pow_dyn[py])
            coe_dhn.append(1j*coe_dyn[py])
            py += 1
        else:
            pow_dhn.append(pow_djn[pj])
            coe_dhn.append(coe_djn[pj] + 1j*coe_dyn[py])
            pj += 1; py += 1
    
    return pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_hndiv, coe_hndiv, pow_dhn, coe_dhn

def rmnMpol_dot(prefactpow, Cpol1, Cpol2):
    #compute unconjugated inner product of two M-type waves represented by A_1mn*Cpol(kr)
    #prefactpow is extra multiplicative factor of (kr)^prefactpow
    intweight = np.zeros(prefactpow+2,dtype=np.complex).tolist()
    intgd = Cpol1*Cpol2
    intpol = po.polyint(intweight+intgd.coef.tolist())
    return prefactpow+3, intpol[prefactpow+3:]

def rmnNpol_dot(prefactpow, Bpol1,Ppol1, Bpol2,Ppol2):
    #compute unconjugated dot product of two N-type waves represented by A_2mn * Bpol(r) + A_3mn * Ppol(r)
    #prefactpow is an extra multiplicative factor of (kr)^prefactpow
    #over spherical domain, returns polynomial coefficients
    intweight = np.zeros(prefactpow+2, dtype=np.complex).tolist() #the +2 represents the radial integral weight function (kr)^2
    intgd = Bpol1*Bpol2 + Ppol1*Ppol2
    intpol = po.polyint(intweight+intgd.coef.tolist())
    return prefactpow+3, intpol[prefactpow+3:] #return an updated prefactor and polynomial coeffs for (kr)^{-prefactor} * integrationresult

def rmnMnormsqr_Taylor(n,k,R, rmnCpol):
    kR = mp.mpf(k*R)
    prefactpow, normpol = rmnMpol_dot(2*n, rmnCpol,rmnCpol)
    return kR**prefactpow * mp.re(po.polyval(kR, normpol)) / k**3

def rmnNnormsqr_Taylor(n,k,R,rmnBpol,rmnPpol):
    #compute norm of spherical wave represented by A_2mn * (kr)^{n-1}*rmnBvec(kr) + A_3mn * (kr)^{n-1}*rmnPvec(kr)
    #no conjugates taken since all the Arnoldi vectors will be purely real
    kR = mp.mpf(k*R)
    prefactpow, normpol = rmnNpol_dot(2*n-2, rmnBpol, rmnPpol, rmnBpol, rmnPpol)
#    print(normpol)
    #print(po.polyval(k*R, normpol))
    return (kR)**prefactpow * mp.re(po.polyval(kR, normpol)) / k**3

def rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, rmnvecM):
    """
    act on wave with radial representation Cpol(kr) by spherical Green's function
    this is for Arnoldi process with the regular M waves
    the leading asymptotice for vecM is r^n
    so rmnv_vecM stores Taylor series representation of (kr)^(-n) * vecM
    rmnRgM stores (kr)^(-n) * RgM, rnImM stores (kr)^(n+1) * ImM
    """
    
    kR = mp.mpf(k*R)
    Mfact_prefactpow, Mfact = rmnMpol_dot(2*n, rmnRgM, rmnvecM)
    Mfact = po.Polynomial(Mfact)
    
    RgMfact_intgd = rnImM*rmnvecM
    RgMfact_intgd = [0] + RgMfact_intgd.coef.tolist()
    RgMfact = po.polyint(RgMfact_intgd,lbnd=kR)
    RgMfact = po.Polynomial(RgMfact)
    
    ###first add on Asym(G) part
    MfactkR = kR**Mfact_prefactpow * po.polyval(kR, Mfact.coef)
    newrmnvecM = 1j*MfactkR * rmnRgM
    #newimag_rmnv_vecM = MfactkR * rmnRgM
    ###then deal with (kr)^(n+2*vecnum+2) part###
    newrmnvecM = newrmnvecM -rnImM * Mfact * po.Polynomial([0,0,1]) + rmnRgM*RgMfact
    #return newreal_rmnv_vecM, newimag_rmnv_vecM #return the real and imaginary part separately
    return newrmnvecM
 
def speedup_Green_Taylor_Arnoldi_RgMmn_oneshot(n,k,R,vecnum,klim=10,Taylor_tol=1e-4,zinv=0.1):
    kR = mp.mpf(k*R)
    pow_jn, coe_jn, pow_yn, coe_yn = get_Taylor_jn_yn(n,kR,klim,Taylor_tol)
    print(len(pow_jn))
    pow_jn=np.array(pow_jn); coe_jn=np.array(coe_jn)
    rmnRgM = mp.one*np.zeros(pow_jn[-1]+1 - n)
    rmnRgM[pow_jn-n] += coe_jn
    rmnRgM = po.Polynomial(rmnRgM)
    
    pow_yn=np.array(pow_yn); coe_yn=np.array(coe_yn)
    rnImM = mp.one*np.zeros(pow_yn[-1]+1 + n+1)
    rnImM[pow_yn+n+1] += coe_yn
    rnImM = po.Polynomial(rnImM)
    
    #plot_rmnMpol(n,rmnRgM.coef, kR*0.01,kR)
    
    norms = []
    unitrmnMpols = []
    
    norms.append(mp.sqrt(rmnMnormsqr_Taylor(n,k,R, rmnRgM)))
    rmnMpol = rmnRgM / norms[0]
    unitrmnMpols.append(rmnMpol)
    
    for i in range(1,vecnum):
        newrmnMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, rmnMpol)
        
        for j in range(len(unitrmnMpols)):
            unitrmnMpol = unitrmnMpols[j]
            prefactpow, coefpol = rmnMpol_dot(2*n, unitrmnMpol, newrmnMpol)
            unitcoef = kR**prefactpow * po.polyval(kR,coefpol) / k**3
            newrmnMpol = newrmnMpol - unitcoef*unitrmnMpol
        
        newrmnMpol = newrmnMpol.cutdeg(rmnRgM.degree())
        
        norms.append(mp.sqrt(rmnMnormsqr_Taylor(n,k,R, newrmnMpol)))
        rmnMpol = newrmnMpol / norms[-1]
        
        #plot_rmnMpol(n, mp_re(rmnMpol.coef), 0.01*kR, kR, pointnum=500)
        unitrmnMpols.append(rmnMpol)
        
    Green_mat = np.zeros((vecnum,vecnum), dtype=np.complex)
    for i in range(vecnum):
        Gi_rmnMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, unitrmnMpols[i])
        prefactpow, Gmatpol = rmnMpol_dot(2*n, unitrmnMpols[i], Gi_rmnMpol)
        Green_mat[i,i] = mp_to_complex(kR**prefactpow * po.polyval(kR, Gmatpol) / k**3)
        
        if i>0:
            Gj_rmnMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, unitrmnMpols[i-1])
            prefactpow, Gmatpol = rmnMpol_dot(2*n, unitrmnMpols[i], Gj_rmnMpol)
            Green_mat[i-1,i] = mp_to_complex(kR**prefactpow * po.polyval(kR,Gmatpol) / k**3)
            Green_mat[i,i-1] = Green_mat[i-1,i]
    
    Gmat2 = np.zeros((vecnum,vecnum),dtype=np.complex)
    for i in range(vecnum):
        for j in range(vecnum):
            Gj_rmnMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM,rnImM,unitrmnMpols[j])
            prefactpow, Gmatpol = rmnMpol_dot(2*n, unitrmnMpols[i], Gj_rmnMpol)
            Gmat2[i,j] = mp_to_complex(kR**prefactpow * po.polyval(kR,Gmatpol) / k**3)
            
    """
    print(Green_mat)
    plt.figure()
    plt.matshow(np.abs(Green_mat))
    plt.show()
    
    Uinv = np.diag(zinv*np.ones(vecnum,dtype=np.complex)) - Green_mat
    U = np.linalg.inv(Uinv)
#    UdU = np.transpose(np.conj(U)) @ U
    print('first row of U matrix')
    print(U[0,:])
    print('norm of U matrix')
    print(np.linalg.norm(U[0,:]))
    print('U matrix visualization')
    plt.figure()
    plt.matshow(np.abs(U))
    plt.show()
    print('speedup norm',norms)
    """
    return Green_mat, Gmat2
    
def plot_RgMmn_g22_g33_g12_g23(n,k,Rmin,Rmax, klim=10,Taylor_tol=1e-4, Rnum=50):
    Rlist = np.linspace(Rmin,Rmax,Rnum)
    absg11list = np.zeros_like(Rlist)
    g22list = np.zeros_like(Rlist); g33list = np.zeros_like(Rlist)
    g12list = np.zeros_like(Rlist); g23list = np.zeros_like(Rlist)
    rholist = np.zeros_like(Rlist)
    for i in range(Rnum):
        print(i)
        R = Rlist[i]
        rholist[i] = mp_rho_M(n, k*R)
        Gmat = speedup_Green_Taylor_Arnoldi_RgMmn_oneshot(n,k,R, 3,klim,Taylor_tol) #vecnum=3 for the elements we need
        absg11list[i] = mp.fabs(Gmat[0,0])
        g22list[i] = mp.re(Gmat[1,1]) #note the indexing off-by-one
        g33list[i] = mp.re(Gmat[2,2])
        g12list[i] = mp.re(Gmat[0,1])
        g23list[i] = mp.re(Gmat[1,2])
    
    normalRlist = Rlist * k / (2*np.pi) #R/lambda list
    plt.figure()
    plt.plot(normalRlist, g22list, '-r', label='$G_{22}$')
    plt.plot(normalRlist, g12list, '--r', label='$G_{12}$')
    plt.plot(normalRlist, g33list, '-b', label='$G_{33}$')
    plt.plot(normalRlist, g23list, '--b', label='$G_{23}$')
    plt.plot(normalRlist, absg11list,'-k', label='$|G_{11}|$')
    plt.plot(normalRlist, rholist, '--k', label='$\\rho_M$')
    
    plt.xlabel('$R/\lambda$',**axisfont)
    plt.title('Green Function matrix elements \n for Arnoldi family radial number n='+str(n),**axisfont)
    plt.legend()
    plt.show()
    

def speedup_Green_Taylor_Arnoldi_step_RgMmn(n,k,R, invchi, rmnRgM, rnImM, unitrmnMpols, Uinv, plotVectors=False):
    kR = mp.mpf(k*R)
    #this method does one extra Arnoldi step, given existing Arnoldi vectors in unitrmnMpols
    #the last entry in the unitPols list is G*unitPols[-2], and len(unitPols) = len(Uinv)+1
    #this setup for most efficient iteration; G*unitPols is only computed once
    #modifies the unitBpols & unitPpols lists on spot
    #U is mpmath matrix, also modified on spot

    #first, begin by orthogonalizing unitPols[-1]
    #use relation for U: U = V^{\dagger*-1} - G
    kR = mp.mpf(k*R)
    unitcoef1 = invchi - Uinv[Uinv.rows-1,Uinv.cols-1]
    unitrmnMpols[-1] = unitrmnMpols[-1] - unitcoef1*unitrmnMpols[-2]
    
    if Uinv.rows>1:
        unitcoef2 = -Uinv[Uinv.rows-2,Uinv.cols-1]
        unitrmnMpols[-1] = unitrmnMpols[-1] - unitcoef2*unitrmnMpols[-3]
    
    norm = mp.sqrt(rmnMnormsqr_Taylor(n,k,R, unitrmnMpols[-1]))
    unitrmnMpols[-1] /= norm
    #since G is tridiagonal under Arnoldi basis this is all the orthogonalization needed
    if plotVectors:
        plot_rmnMpol(n, mp_re(unitrmnMpols[-1].coef), 0.01*kR, kR, pointnum=500)
    
    #get new vector
    newrmnMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, unitrmnMpols[-1])
    newrmnMpol = newrmnMpol.cutdeg(rmnRgM.degree())
    
    Uinv.rows = Uinv.rows+1; Uinv.cols = Uinv.cols+1 #expand Uinv
    prefactpow, Udiagpol = rmnMpol_dot(2*n, unitrmnMpols[-1], newrmnMpol)
    Uinv[Uinv.rows-1,Uinv.cols-1] = invchi - kR**prefactpow * po.polyval(kR, Udiagpol) / k**3
    
    prefactpow, Uoffdpol = rmnMpol_dot(2*n, unitrmnMpols[-2], newrmnMpol)
    Uinv[Uinv.rows-2,Uinv.cols-1] = -kR**prefactpow * po.polyval(kR, Uoffdpol) / k**3
    Uinv[Uinv.rows-1,Uinv.cols-2] = Uinv[Uinv.rows-2, Uinv.cols-1]
    
    unitrmnMpols.append(newrmnMpol)


def speedup_Green_Taylor_Arnoldi_RgMmn_Uconverge(n,k,R, klim=10, Taylor_tol=1e-8, invchi=0.1, Unormtol=1e-4, veclim=3, delveclim=2, plotVectors=False):
    #sets up the Arnoldi matrix and associated unit vector lists for any given RgM
    
    kR = mp.mpf(k*R)
    pow_jn, coe_jn, pow_yn, coe_yn = get_Taylor_jn_yn(n,kR,klim,Taylor_tol)
    #print(len(pow_jn))
    pow_jn=np.array(pow_jn); coe_jn=np.array(coe_jn)
    rmnRgM = mp.one*np.zeros(pow_jn[-1]+1 - n)
    rmnRgM[pow_jn-n] += coe_jn
    rmnRgM = po.Polynomial(rmnRgM)
    
    pow_yn=np.array(pow_yn); coe_yn=np.array(coe_yn)
    rnImM = mp.one*np.zeros(pow_yn[-1]+1 + n+1)
    rnImM[pow_yn+n+1] += coe_yn
    rnImM = po.Polynomial(rnImM)
    
    if plotVectors:
        plot_rmnMpol(n, rmnRgM.coef, 0.01*kR, kR)
    
    unitrmnMpols=[]
    RgMnorm = mp.sqrt(rmnMnormsqr_Taylor(n,k,R, rmnRgM))
    rmnMpol = rmnRgM / RgMnorm
    unitrmnMpols.append(rmnMpol)
    
    rmnGMpol = rmnGreen_Taylor_Mmn_vec(n,k,R, rmnRgM, rnImM, rmnMpol)
    rmnGMpol = rmnGMpol.cutdeg(rmnRgM.degree())
    prefactpow, Upol = rmnMpol_dot(2*n, rmnMpol, rmnGMpol)
    Uinv = mp.matrix([[invchi - kR**prefactpow * po.polyval(kR, Upol) / k**3]])
    unitrmnMpols.append(rmnGMpol)  #set up beginning of Arnoldi iteration
    
    b = mp.matrix([mp.one])
    prevUnorm = 1 / Uinv[0,0]
    
    i=1
    while i<veclim:
        speedup_Green_Taylor_Arnoldi_step_RgMmn(n,k,R, invchi, rmnRgM, rnImM, unitrmnMpols, Uinv, plotVectors=plotVectors)
        i += 1
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            b.rows = i
            x = mp.lu_solve(Uinv, b)
            Unorm = mp.norm(x)
            if np.abs(prevUnorm-Unorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
            #else: print(x)
    
    if veclim==1:
        x = mp.one / Uinv[0,0]
        
    """
    print(x)
    print(mp.norm(x))
    plt.figure()
    plt.matshow(np.abs(np.array((Uinv**-1).tolist(),dtype=np.complex)))
    plt.show()
    #EUdUinv = mpmath.eigh(Uinv.transpose_conj()*Uinv, eigvals_only=True)
    #print(EUdUinv)
    """
    #returns everything necessary to potentially extend size of Uinv matrix later
    return rmnRgM, rnImM, unitrmnMpols, Uinv

def rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol, rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, rmnvecBpol, rmnvecPpol):
    """
    act on the wave with radial representation Bpol(kr), Ppol(kr) by the spherical Green's fcn
    central subroutine for Arnoldi iteration starting from RgN
    rmnRgN_Bpol, rmnRgN_Ppol stores Taylor series representation of (kr)^{-(n-1)}*RgN(kr) with np polynomials
    rnImN_Bpol, rnImN_Ppol stores Taylor series representation of (kr)^{n+2} * Im{N(kr)}
    the real part of N(kr) is just RgN(kr)
    rmnvecBpol, rmnvecPpol store Taylor series representations of (kr)^{-(n-1)}*vec(kr), where vec is the current Arnoldi iterate
    the prefactors are there to efficiently use the numpy polynomial package,
    avoiding carrying a long list of zeros leading to large polynomials and slow running time
    N has radial dependence with negative powers so the prefactor lets use np polynomials
    just need to divide results after multiplication of N by (kr)^{n+2} 
    """
    kR = mp.mpf(k*R)
    ti = time.time()
    Nfact_prefactpow, Nfact = rmnNpol_dot(2*n-2, rmnRgN_Bpol, rmnRgN_Ppol, rmnvecBpol, rmnvecPpol)
    #Nfact_prefactpow should be 2*n+1
    Nfact = po.Polynomial(Nfact)
    #Nfact = po.Polynomial(1j*Npol_dot(RgN_Bpol,RgN_Ppol, vecBpol,vecPpol))
    
    RgNfact_intgd = rnImN_Bpol*rmnvecBpol + rnImN_Ppol*rmnvecPpol
    RgNfact_intgd = [0,0] + RgNfact_intgd.coef.tolist()
    #print('1/r term size',RgNfact_intgd[2]) #check that the 1/r term of integrand is indeed 0 up to floating point error
    #print('r term size',RgNfact_intgd[4])
    RgNfact_intgd = po.Polynomial(RgNfact_intgd[3:]) #n+2-(n-1) divide by (kr)^{3} to get true intgd

    #print(time.time()-ti,'in GTNv, 1')
#    print(len(RgNfact_intgd))
    RgNfact = po.Polynomial(po.polyint(RgNfact_intgd.coef,lbnd=kR))

    #print(time.time()-ti,'in GTNv, 2')
    
    ######first add on the Asym(G) part
    NfactkR = kR**Nfact_prefactpow * po.polyval(kR, Nfact.coef)
    newrmnBpol = 1j*NfactkR*rmnRgN_Bpol
    newrmnPpol = 1j*NfactkR*rmnRgN_Ppol

    #print(time.time()-ti,'in GTNv, 3')
    ######then add on the Sym(G) part
    newrmnBpol = newrmnBpol - Nfact*rnImN_Bpol
    newrmnPpol = newrmnPpol - Nfact*rnImN_Ppol
    newrmnBpol = newrmnBpol + RgNfact*rmnRgN_Bpol
    newrmnPpol = newrmnPpol + RgNfact*rmnRgN_Ppol - rmnvecPpol #the subtraction at end is delta fcn contribution
    #print(time.time()-ti,'in GTNv, 4')
    #print(len(newrmnBpol),len(newrmnPpol),len(RgNfact),len(Nfact),len(rmnRgN_Bpol),len(rmnRgN_Ppol),len(rnImN_Bpol), len(rnImN_Ppol))
    return newrmnBpol, newrmnPpol

def plot_GNmn_image(n,k,R, rmnvecBpol, rmnvecPpol, klim=30, Taylor_tol=1e-4):
    kR = mp.mpf(k*R)
    pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_yndiv, coe_yndiv, pow_dyn, coe_dyn = get_Taylor_jndiv_djn_yndiv_dyn(n,kR,klim,tol=Taylor_tol)
    print(len(pow_jndiv))
    nfac = mp.sqrt(n*(n+1))
    pow_jndiv = np.array(pow_jndiv); coe_jndiv = np.array(coe_jndiv)
    pow_djn = np.array(pow_djn); coe_djn = np.array(coe_djn)
    rmnRgN_Bpol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmnRgN_Ppol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmnRgN_Bpol[pow_jndiv-(n-1)] += coe_jndiv
    rmnRgN_Bpol[pow_djn-(n-1)] += coe_djn
    rmnRgN_Ppol[pow_jndiv-(n-1)] += coe_jndiv
    rmnRgN_Ppol *= nfac
    
    rmnRgN_Bpol = po.Polynomial(rmnRgN_Bpol); rmnRgN_Ppol = po.Polynomial(rmnRgN_Ppol)
    
    rnImN_Bpol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    rnImN_Ppol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    pow_yndiv = np.array(pow_yndiv); pow_dyn = np.array(pow_dyn)
    rnImN_Bpol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Bpol[(n+2)+pow_dyn] += coe_dyn
    rnImN_Ppol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Ppol *= nfac
    
    rnImN_Bpol = po.Polynomial(rnImN_Bpol); rnImN_Ppol = po.Polynomial(rnImN_Ppol)
    
    rmnBimage, rmnPimage = rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol, rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, rmnvecBpol, rmnvecPpol)
    print('real part')
    plot_rmnNpol(n, mp_re(rmnBimage.coef), mp_re(rmnPimage.coef), kR*0.01, kR)
    
    print('imag part')
    plot_rmnNpol(n, mp_im(rmnBimage.coef), mp_im(rmnPimage.coef), kR*0.01, kR)

def speedup_Green_Taylor_Arnoldi_RgNmn_oneshot(n,k,R,vecnum,klim=10,Taylor_tol=1e-4,zinv=0.1):
    kR = mp.mpf(k*R)
    pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_yndiv, coe_yndiv, pow_dyn, coe_dyn = get_Taylor_jndiv_djn_yndiv_dyn(n,kR,klim,tol=Taylor_tol)
    print(len(pow_jndiv))
    
    nfac = mp.sqrt(n*(n+1))
    pow_jndiv = np.array(pow_jndiv); coe_jndiv = np.array(coe_jndiv)
    pow_djn = np.array(pow_djn); coe_djn = np.array(coe_djn)
    rmnRgN_Bpol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmnRgN_Ppol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmnRgN_Bpol[pow_jndiv-(n-1)] += coe_jndiv
    rmnRgN_Bpol[pow_djn-(n-1)] += coe_djn
    rmnRgN_Ppol[pow_jndiv-(n-1)] += coe_jndiv
    rmnRgN_Ppol *= nfac
    
    rmnRgN_Bpol = po.Polynomial(rmnRgN_Bpol); rmnRgN_Ppol = po.Polynomial(rmnRgN_Ppol)
    
    rnImN_Bpol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    rnImN_Ppol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    pow_yndiv = np.array(pow_yndiv); pow_dyn = np.array(pow_dyn)
    rnImN_Bpol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Bpol[(n+2)+pow_dyn] += coe_dyn
    rnImN_Ppol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Ppol *= nfac
    
    rnImN_Bpol = po.Polynomial(rnImN_Bpol); rnImN_Ppol = po.Polynomial(rnImN_Ppol)
#    print(len(rnImN_Bpol), pow_yndiv[0],pow_yndiv[-1])
    plot_rmnNpol(n,rmnRgN_Bpol.coef, rmnRgN_Ppol.coef, kR*0.01, kR)
    
    norms = []
    unitrmnBpols = []; unitrmnPpols = []
    
    norms.append(mp.sqrt(rmnNnormsqr_Taylor(n,k,R, rmnRgN_Bpol, rmnRgN_Ppol)))
    rmnBpol = rmnRgN_Bpol / norms[0]; rmnPpol = rmnRgN_Ppol / norms[0]
    unitrmnBpols.append(rmnBpol); unitrmnPpols.append(rmnPpol)
    
    
    for i in range(1,vecnum):
        print(i)
#        print(len(rmnBpol))
        newrmnBpol, newrmnPpol = rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol, rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, rmnBpol, rmnPpol)
        #plot_Npol(mp_im(newrmnBpol.coef),mp_im(newrmnPpol.coef), kR*0.01, kR)
        #plot_rmnNpol(n, mp_re(newrmnBpol.coef),mp_re(newrmnPpol.coef), kR*0.01, kR,pointnum=500)
        #print('the image of vec',i,' has leading orders for non-radial', newrmnBpol.coef[0], newrmnBpol.coef[1], newrmnBpol.coef[2])
        #print('the image of vec',i,' has leading orders for radial', newrmnPpol.coef[0], newrmnPpol.coef[1], newrmnPpol.coef[2])
        #newBpol = po.Polynomial(np.real(newBpol.coef)); newPpol = po.Polynomial(np.real(newPpol.coef))
        
        for j in range(len(unitrmnBpols)):
            unitrmnBpol = unitrmnBpols[j]; unitrmnPpol = unitrmnPpols[j]
            prefactpow, coefpol = rmnNpol_dot(2*n-2, unitrmnBpol,unitrmnPpol, newrmnBpol,newrmnPpol)
            unitcoef = kR**prefactpow * po.polyval(kR,coefpol) / k**3 #no conjugation since the unit vectors are all real
            newrmnBpol = newrmnBpol - unitcoef*unitrmnBpol
            newrmnPpol = newrmnPpol - unitcoef*unitrmnPpol
            
        newrmnBpol = newrmnBpol.cutdeg(rmnRgN_Bpol.degree()*1)
        newrmnPpol = newrmnPpol.cutdeg(rmnRgN_Ppol.degree()*1)
        
        norms.append(mp.sqrt(rmnNnormsqr_Taylor(n,k,R, newrmnBpol, newrmnPpol)))
        rmnBpol = newrmnBpol / norms[-1]; rmnPpol = newrmnPpol / norms[-1]
        #print(' vec',i,' has leading orders for non-radial', rmnBpol.coef[0], rmnBpol.coef[1], rmnBpol.coef[2])
        #print('vec',i,' has leading orders for radial', rmnPpol.coef[0], rmnPpol.coef[1], rmnPpol.coef[2])
        plot_rmnNpol(n, mp_re(rmnBpol.coef),mp_re(rmnPpol.coef), kR*0.01, kR,pointnum=500)
        
        unitrmnBpols.append(rmnBpol); unitrmnPpols.append(rmnPpol)
        
    Green_mat = np.zeros((vecnum,vecnum),dtype=np.complex)
    for i in range(vecnum):
        Gi_rmnBpol, Gi_rmnPpol = rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol,rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols[i],unitrmnPpols[i])
        prefactpow, Gmatpol = rmnNpol_dot(2*n-2, unitrmnBpols[i],unitrmnPpols[i], Gi_rmnBpol,Gi_rmnPpol)
        Green_mat[i,i] = mp_to_complex(kR**prefactpow * po.polyval(kR, Gmatpol) / k**3)
        if i>0:
            Gj_rmnBpol, Gj_rmnPpol = rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol,rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols[i-1],unitrmnPpols[i-1])
            prefactpow, Gmatpol = rmnNpol_dot(2*n-2, unitrmnBpols[i],unitrmnPpols[i], Gj_rmnBpol,Gj_rmnPpol)
            Green_mat[i-1,i] = mp_to_complex(kR**prefactpow * po.polyval(kR, Gmatpol) / k**3)
            Green_mat[i,i-1] = Green_mat[i-1,i]
    
    print(Green_mat)
    plt.figure()
    plt.matshow(np.abs(Green_mat))
    plt.title('Greens Function')
    plt.show()
    
    Uinv = np.diag(zinv*np.ones(vecnum,dtype=np.complex)) - Green_mat
    #Uinv = - Green_mat
    SymU = np.real(Uinv+np.conjugate(Uinv.T))/2
    print(SymU)
    plt.figure()
    plt.matshow(SymU)
    plt.title('SymU')
    plt.show()
    """
    b = np.zeros(vecnum,dtype=np.complex)
    b[0] = 1
    x = np.linalg.solve(Uinv,b)
    print(x)
    print(np.linalg.norm(x))
    
    U = np.linalg.inv(Uinv)
#    UdU = np.transpose(np.conj(U)) @ U
    print('first row of U matrix')
    print(U[0,:])
    print('norm of U matrix')
    print(np.linalg.norm(U[0,:]))
    print('U matrix visualization')
    plt.figure()
    plt.matshow(np.abs(U))
    plt.show()
    print('speedup norm',norms)
    """

def speedup_Green_Taylor_Arnoldi_step_RgNmn(n,k,R,invchi, rmnRgN_Bpol, rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols, unitrmnPpols, Uinv, plotVectors=False):
    #this method does one extra Arnoldi step, given existing Arnoldi vectors in unitBpols and unitPpols
    #the last entry in the unitPols list is G*unitPols[-2], and len(unitPols) = len(Uinv)+1
    #this setup for most efficient iteration; G*unitPols is only computed once
    #modifies the unitBpols & unitPpols lists on spot
    #U is mpmath matrix, also modified on spot

    #first, begin by orthogonalizing unitPols[-1]
    #use relation for U: U = V^{\dagger*-1} - G
    kR = mp.mpf(k*R)
    unitcoef1 = invchi-Uinv[Uinv.rows-1,Uinv.cols-1]
    unitrmnBpols[-1] = unitrmnBpols[-1] - unitcoef1*unitrmnBpols[-2]
    unitrmnPpols[-1] = unitrmnPpols[-1] - unitcoef1*unitrmnPpols[-2]
    if Uinv.rows>1:
        unitcoef2 = -Uinv[Uinv.rows-2,Uinv.cols-1] #off-diagonal, no V^{\dagger*-1} contribution
        unitrmnBpols[-1] = unitrmnBpols[-1] - unitcoef2*unitrmnBpols[-3]
        unitrmnPpols[-1] = unitrmnPpols[-1] - unitcoef2*unitrmnPpols[-3]
    
    norm = mp.sqrt(rmnNnormsqr_Taylor(n,k,R, unitrmnBpols[-1], unitrmnPpols[-1]))
    unitrmnBpols[-1] /= norm; unitrmnPpols[-1] /= norm
    #since G is tridiagonal under Arnoldi basis this is all the orthogonalization needed
    if plotVectors:
        plot_rmnNpol(n, mp_re(unitrmnBpols[-1].coef),mp_re(unitrmnPpols[-1].coef), kR*0.01, kR,pointnum=500)
    
    #get new vector
    newrmnBpol, newrmnPpol = rmnGreen_Taylor_Nmn_vec(n,k,R, rmnRgN_Bpol, rmnRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols[-1], unitrmnPpols[-1])
    newrmnBpol = newrmnBpol.cutdeg(rmnRgN_Bpol.degree()*1)
    newrmnPpol = newrmnPpol.cutdeg(rmnRgN_Ppol.degree()*1)
    
    Uinv.rows = Uinv.rows+1; Uinv.cols = Uinv.cols+1 #expand Uinv
    #U = V^{\dagger*-1} - G
    prefactpow, Udiagpol = rmnNpol_dot(2*n-2, unitrmnBpols[-1],unitrmnPpols[-1], newrmnBpol,newrmnPpol)
    Uinv[Uinv.rows-1,Uinv.cols-1] = invchi - kR**prefactpow * po.polyval(kR, Udiagpol) / k**3
    
    prefactpow, Uoffdpol = rmnNpol_dot(2*n-2, unitrmnBpols[-2],unitrmnPpols[-2], newrmnBpol,newrmnPpol)
    Uinv[Uinv.rows-2,Uinv.cols-1] = -kR**prefactpow * po.polyval(kR, Uoffdpol) / k**3
    Uinv[Uinv.rows-1,Uinv.cols-2] = Uinv[Uinv.rows-2,Uinv.cols-1]
    
    unitrmnBpols.append(newrmnBpol); unitrmnPpols.append(newrmnPpol)
    

def speedup_Green_Taylor_Arnoldi_RgNmn_Uconverge(n,k,R, klim=10,Taylor_tol=1e-8,invchi=0.1, Unormtol=1e-4, veclim=3, delveclim=2, plotVectors=False):
    #sets up the Arnoldi matrix and associated unit vector lists for any given RgN
    #ti = time.time()
    
    kR = mp.mpf(k*R)
    pow_jndiv, coe_jndiv, pow_djn, coe_djn, pow_yndiv, coe_yndiv, pow_dyn, coe_dyn = get_Taylor_jndiv_djn_yndiv_dyn(n,kR,klim,tol=Taylor_tol)
    #print(time.time()-ti,'0')
    
    #print(len(pow_jndiv))
    pow_jndiv=np.array(pow_jndiv); pow_djn=np.array(pow_djn)
    nfac = mp.sqrt(n*(n+1))
    rmRgN_Bpol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmRgN_Ppol = mp.one*np.zeros(pow_jndiv[-1]+1 - (n-1))
    rmRgN_Bpol[pow_jndiv-(n-1)] += coe_jndiv
    rmRgN_Bpol[pow_djn-(n-1)] += coe_djn
    rmRgN_Ppol[pow_jndiv-(n-1)] += coe_jndiv
    rmRgN_Ppol *= nfac
    
    rmRgN_Bpol = po.Polynomial(rmRgN_Bpol); rmRgN_Ppol = po.Polynomial(rmRgN_Ppol)
    
    rnImN_Bpol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    rnImN_Ppol = mp.one*np.zeros(pow_yndiv[-1]+1+(n+2),dtype=np.complex)
    pow_yndiv = np.array(pow_yndiv); pow_dyn = np.array(pow_dyn)
    rnImN_Bpol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Bpol[(n+2)+pow_dyn] += coe_dyn
    rnImN_Ppol[(n+2)+pow_yndiv] += coe_yndiv
    rnImN_Ppol *= nfac
    
    rnImN_Bpol = po.Polynomial(rnImN_Bpol); rnImN_Ppol = po.Polynomial(rnImN_Ppol)
    
    if plotVectors:
        plot_rmnNpol(n, rmRgN_Bpol.coef, rmRgN_Ppol.coef, kR*0.01, kR)
    
    unitrmnBpols = []; unitrmnPpols = []
    #print(time.time()-ti,'1')
    RgNnorm = mp.sqrt(rmnNnormsqr_Taylor(n, k,R, rmRgN_Bpol, rmRgN_Ppol))
    rmnBpol = rmRgN_Bpol / RgNnorm; rmnPpol = rmRgN_Ppol / RgNnorm
    unitrmnBpols.append(rmnBpol); unitrmnPpols.append(rmnPpol)
    #print(time.time()-ti,'2')
    rmnGBpol, rmnGPpol = rmnGreen_Taylor_Nmn_vec(n,k,R, rmRgN_Bpol, rmRgN_Ppol, rnImN_Bpol,rnImN_Ppol, rmnBpol, rmnPpol)
    rmnGBpol = rmnGBpol.cutdeg(rmRgN_Bpol.degree())
    rmnGPpol = rmnGPpol.cutdeg(rmRgN_Bpol.degree())
    #print(time.time()-ti,'3')
    prefactpow, Upol = rmnNpol_dot(2*n-2, rmnBpol,rmnPpol, rmnGBpol,rmnGPpol)
    Uinv = mp.matrix([[invchi - kR**prefactpow * po.polyval(kR, Upol) / k**3]])
    unitrmnBpols.append(rmnGBpol); unitrmnPpols.append(rmnGPpol)  #set up beginning of Arnoldi iteration
    #print(time.time()-ti,'4')
    b = mp.matrix([mp.one])
    prevUnorm = 1 / Uinv[0,0]
    
    i=1
    while i<veclim:
        speedup_Green_Taylor_Arnoldi_step_RgNmn(n,k,R,invchi, rmRgN_Bpol,rmRgN_Ppol,rnImN_Bpol,rnImN_Ppol, unitrmnBpols,unitrmnPpols,Uinv, plotVectors=plotVectors)
        i += 1
        if i==veclim:
            #solve for first column of U and see if its norm has converged
            b.rows = i
            x = mp.lu_solve(Uinv,b)
            Unorm = mp.norm(x)
            if np.abs(prevUnorm-Unorm) > np.abs(Unorm)*Unormtol:
                veclim += delveclim
                prevUnorm = Unorm
            #else: print(x)
    
    if veclim==1:
        x = mp.one / Uinv[0,0]
    
    """
    #print(x)
    print(mp.norm(x))
    plt.figure()
    plt.matshow(np.abs(np.array((Uinv**-1).tolist(),dtype=np.complex)))
    plt.show()
    #EUdUinv = mpmath.eigh(Uinv.transpose_conj()*Uinv, eigvals_only=True)
    #print(EUdUinv)
    #print(type(EUdUinv))
    """
    #print(time.time()-ti,'5')
    #returns everything necessary to potentially extend size of Uinv matrix later
    return rmRgN_Bpol, rmRgN_Ppol, rnImN_Bpol, rnImN_Ppol, unitrmnBpols, unitrmnPpols, Uinv


"""
the following code extracts a numerical matrix representation of the Green's function
out of U.
Note in the following code written much later than the previous code,
U is now the complex conjugate of the old U used in the previous part of this script
"""
def get_G_from_U(U, chi):
    n = U.rows
    Vinvdagger = mp.eye(n)*mp.one/mp.conj(chi)
    Gdagger = Vinvdagger - U
    return Gdagger.H
