#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 14:48:19 2020

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
from dualbound.Arnoldi.spherical_multiRegion_Green_Arnoldi import spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge, spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge
from dualbound.Lagrangian.spatialProjopt_Zops_numpy import get_ZTT_mineig, get_inc_ZTT_mineig
from dualbound.Lagrangian.spatialProjopt_feasiblept import spatialProjopt_find_feasiblept
from dualbound.Lagrangian.spatialProjopt_dualgradHess_fullSvec_numpy import get_inc_spatialProj_dualgradHess_fullS, get_incgrad_from_grad, get_incHess_from_Hess
from dualbound.Optimization.modSource_opt import modS_opt


def get_planewave_sphere_multiRegion_S1list_Ulists_Projlists_numpy(k, RPlist, chi, 
                                                                   mpdps=60, normtol = 1e-8, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True):
    
    mp.dps=mpdps
    spherenorm = 4*np.pi*RPlist[-1]**3 / 3 #the norm of the unity amplitude planewave source over the spherical bounding domain
    sumnorm = 0.0
    invchi = 1.0/chi
    
    S1list = []
    Plistlist = []
    UPlistlist = [] #lists for the source vector, Umatrix representation and U*Proj_i matrix representations to be constructed
    subbasis_indlistM = []; subbasis_indlistN = [] #stores the indices corresponding to each regional subbasis in the matrix representations
    klim_update = 0 #minimum suggested klim for Arnoldi process of the inner spherical region to converge; computed to help with future sweep calls
    l=1
    while spherenorm-sumnorm>normtol*spherenorm:
        if l==1:
            Mklim = Minitklim; Nklim = Ninitklim
        else:
            #the first subregion is the inner sphere; its Arnoldi process uses the old Taylor Arnoldi code
            Mklim = 2*(subbasis_indlistM[1]-subbasis_indlistM[0])+5
            Nklim = 2*(subbasis_indlistN[1]-subbasis_indlistN[0])+5
                
        ############################M wave component##############################
        Gmat, Uconj, RgMnormlist, subbasis_indlistM, fullrgrid, All_fullr_unitMvecs = spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge(l,k,RPlist, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Mklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
        U = Uconj.conjugate()
        
        if includeI: #now extend Plistlist and UPlistlist
            Plist = [np.eye(U.shape[0])]
            UPlist = [U]
        else:
            Plist = []
            UPlist = []
            
        for i in range(len(RPlist)):
            Proj = np.zeros_like(U)
            subbasis_head = subbasis_indlistM[i]; subbasis_tail = subbasis_indlistM[i+1]
            Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
            UProj = U @ Proj
            Plist.append(Proj)
            UPlist.append(UProj)
            
        Plistlist.append(Plist)
        UPlistlist.append(UPlist)
        
        ############################N wave component##############################
        Gmat, Uconj, RgNnormlist, subbasis_indlistN, fullrgrid, All_fullr_unitBvecs,All_fullr_unitPvecs = spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge(l,k,RPlist, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Nklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
        U = Uconj.conjugate()
        
        if includeI: #now extend Plistlist and UPlistlist
            Plist = [np.eye(U.shape[0])]
            UPlist = [U]
        else:
            Plist = []
            UPlist = []
            
        for i in range(len(RPlist)):
            Proj = np.zeros_like(U)
            subbasis_head = subbasis_indlistN[i]; subbasis_tail = subbasis_indlistN[i+1]
            Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
            UProj = U @ Proj
            Plist.append(Proj)
            UPlist.append(UProj)
            
        Plistlist.append(Plist)
        UPlistlist.append(UPlist)
        
        #calculate the RgM, RgN component coefficients of the planewave and distribute it over the regional subbases
        RgNe_coef = 2 * (1j)**((l+1)%4) * np.sqrt((2*l+1)*np.pi/2) #the factor of 1/2 in the sqrt is the norm of the odd/even vector spherical harmonics
        RgMo_coef = RgNe_coef * 1j
        
        #construct RgM source vector
        M_S1vec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistM)-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            M_S1vec[subbasis_indlistM[i]] = np.complex(RgMo_coef*RgMnormlist[i])
        S1list.append(M_S1vec)
        
        #construct RgN source vector
        N_S1vec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistN)-1):
            N_S1vec[subbasis_indlistN[i]] = np.complex(RgNe_coef*RgNnormlist[i])
        S1list.append(N_S1vec)
        
        sosqr = np.real(np.vdot(M_S1vec,M_S1vec)+np.vdot(N_S1vec,N_S1vec))
        sumnorm += sosqr
        print('sumnorm', sumnorm, 'spherenorm', spherenorm)
        if l==1:
            klim_update = 5 + 2*max(subbasis_indlistM[1]-subbasis_indlistM[0], subbasis_indlistN[1]-subbasis_indlistN[0])
        l += 1
    
    return S1list, Plistlist, UPlistlist, klim_update


def get_planewave_ZTS_Slist(Lags, S1list, Plistlist):
    ZTS_Slist = []
    for mode in range(len(S1list)):
        Plist = Plistlist[mode]
        ZTS = np.zeros_like(Plist[0], dtype=np.complex)
        for i in range(len(Plist)):
            ZTS += (Lags[2*i]+1j*Lags[2*i+1])*Plist[i].conj().T/2
            
        ZTS_S = ZTS @ S1list[mode] + (1j/2)*S1list[mode] #for planewave, S2 = S1
        ZTS_Slist.append(ZTS_S)
    return ZTS_Slist

def get_planewave_gradZTS_Slist(Lags, S1list, Plistlist):
    gradZTS_Slist = []
    for mode in range(len(S1list)):
        Plist = Plistlist[mode]
        gradZTS_S = []
        for i in range(len(Plist)):
            P_S = Plist[i] @ S1list[mode]
            gradZTS_S.append(P_S/2)
            gradZTS_S.append(1j*P_S/2)
        gradZTS_Slist.append(gradZTS_S)
    return gradZTS_Slist
        

def get_planewave_spherical_multiRegion_Psca_numpy(k,RPlist,chi,
                                                   initLags=None, incPIm=False, incRegions=None, mpdps=60, normtol = 1e-8, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6,
                                                   opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    
    S1list, Plistlist, UPlistlist, klim_update = get_planewave_sphere_multiRegion_S1list_Ulists_Projlists_numpy(k, RPlist, chi, 
                                                                                                                mpdps=60, normtol = 1e-8, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True)
    
    modenum = len(S1list)
    zinv = np.imag(1/np.conj(chi))
    
    Olist = []
    for mode in range(modenum):
        Olist.append(np.eye(Plistlist[mode][0].shape[0])*zinv)
    
    Lagnum = 2 + 2*len(RPlist)
    """
    2 is for the original Re+Im global constraint; then there's the projection versions
    because we always include the original global constraints over the entire region,
    can't include all regional constraints because that would cause the constraints to be linearly dependent
    default is to include all regional constraints except that of the inner shell
    """
    
    include=[True]*Lagnum
    if incRegions is None: #use default region inclusion: every region except the innermost
        include[2] = False; include[3] = False
    else:
        for i in range(len(incRegions)):
            include[2+2*i] = incRegions[i]; include[3+2*i] = incRegions[i]
    if not incPIm:
        for i in range(1,len(RPlist)+1): #i starts from 1 because we always want to keep the original Im constraint
            include[2*i+1] = False
    
    if initLags is None:
        Lags = spatialProjopt_find_feasiblept(Lagnum, include, Olist, UPlistlist)
    else:
        Lags = initLags.copy()
        Lags[1] = np.abs(Lags[1])+0.01
    
    validLagsfunc = lambda L: get_ZTT_mineig(L, Olist, UPlistlist, eigvals_only=True)[1] #[1] because we only need the minimum eigenvalue part of the returned values
    while validLagsfunc(Lags)<0:
        print('zeta', Lags[1])
        Lags[1] *= 1.5 #enlarge zeta until we enter domain of validity
    print('initial Lags', Lags)
    
    ZTS_Slistfunc = lambda L, Slist: get_planewave_ZTS_Slist(L, Slist, Plistlist)
    gradZTS_Slistfunc = lambda L, Slist: get_planewave_gradZTS_Slist(L, Slist, Plistlist)
    dgHfunc = lambda dof, dofgrad, dofHess, Slist, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fullS(dof, dofgrad, dofHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, get_grad=get_grad, get_Hess=get_Hess)
    mineigfunc = lambda dof, eigvals_only=False: get_inc_ZTT_mineig(dof, include, Olist, UPlistlist, eigvals_only=eigvals_only)

    #check_spatialProjopt_gradHess(Lags, S1list, dgHfunc, include=include)
    #return 0

    opt_incLags, opt_incgrad, opt_dual, opt_obj = modS_opt(Lags[include], S1list, dgHfunc, mineigfunc, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
    optLags = np.zeros(Lagnum)
    optgrad = np.zeros(Lagnum)
    optLags[include] = opt_incLags
    optgrad[include] = opt_incgrad
    print('the remaining constraint violations')
    print(optgrad)
    #prefactors from physics
    #the effective cross section enhancement is scattering power (k*opt_dual/(2*Z)) / unity amplitude Poynting vector (1/2Z) / geometric cross section (pi*R**2)
    return optLags, k*opt_dual/np.pi/RPlist[-1]**2, k*opt_obj/np.pi/RPlist[-1]**2, klim_update


def check_spatialProjopt_gradHess(Lags, S1list, dgHfunc, include=None):
    
    dualfunc = lambda incL: dgHfunc(incL, np.array([]), np.array([]), S1list, get_grad=False, get_Hess=False)
    Lagnum = len(Lags)
    incLagnum = np.sum(include)
    incgrad = np.zeros(incLagnum)
    incHess = np.zeros((incLagnum,incLagnum))
    
    P0 = dgHfunc(Lags[include], incgrad, incHess, S1list, get_grad=True, get_Hess=True)
    
    grad = np.zeros(Lagnum); grad[include] = incgrad[:]
    Hess = np.zeros((Lagnum,Lagnum)); Hess[np.ix_(include,include)] = incHess[:,:]
    #P0 = get_twofield_spatialProj_dualgrad(Lags, grad, S_list, O1list, O2list, Plistlist, UPlistlist, include=include)
    print('P0 is', P0)
    
    step = 1e-3
    
    for i in range(Lagnum):
        if include[i]:
            deltai = step*np.abs(Lags[i])
            Lags[i] += deltai
            P1 = dualfunc(Lags[include])
            Lags[i] -= deltai
            fdgrad = (P1-P0)/deltai
            print('for multiplier',i,'the calculated gradient and fd estimate are', grad[i], fdgrad)
    
    for i in range(Lagnum):
        if not include[i]:
            continue
        deltai = step*np.abs(Lags[i])
        piLags = Lags.copy(); piLags[i] += deltai
        miLags = Lags.copy(); miLags[i] -= deltai
        
        fdsqri = (dualfunc(piLags[include])-2*P0+dualfunc(miLags[include]))/deltai**2
        print('for multiplier',i, 'the calculated 2nd derivative and fd estimate are', Hess[i,i], fdsqri)
        
        for j in range(i+1, Lagnum):
            if not include[j]:
                continue
            deltaj = step*np.abs(Lags[j])
            pipjLags = Lags.copy(); pipjLags[i] += deltai; pipjLags[j] += deltaj
            pimjLags = Lags.copy(); pimjLags[i] += deltai; pimjLags[j] -= deltaj
            mipjLags = Lags.copy(); mipjLags[i] -= deltai; mipjLags[j] += deltaj
            mimjLags = Lags.copy(); mimjLags[i] -= deltai; mimjLags[j] -= deltaj
            fd2ij = (dualfunc(pipjLags[include])-dualfunc(pimjLags[include])-dualfunc(mipjLags[include])+dualfunc(mimjLags[include]))/(4*deltai*deltaj)
            print('for multipliers',i,j,'the calculated cross derivative and fd estimate are', Hess[i,j], fd2ij)
    
    testSym = Hess - Hess.T
    print('test Hessian symmetry', np.linalg.norm(testSym))
    
    incHeigw = np.linalg.eigvalsh(incHess)
    print('test Hess PD', incHeigw)
    
    
def sweep_Psca_spherical_planewave_spatialProjlists_varyR(k, Rlist, chi, RPfaclist, normtol = 1e-8, incPIm=False, mpdps=60, maxveclim=40, Minitklim=25, Ninitklim=25, Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, check_iter_period=20,
                                                          filename=None, feedforward=True):

    flag = False
    if not (filename is None):
        flag = True
        outR = open(filename+'_Rlist.txt','w'); outdual = open(filename+'_duallist.txt','w')
        outobj = open(filename+'_objlist.txt','w'); outasym = open(filename+'_asymlist.txt','w')
    
    duallist=[]; objlist=[]; currentRlist=[]
    Lags = None
    klim=Minitklim #start with no initial condition
    
    for i in range(len(Rlist)):
        R = Rlist[i]
        #the external program determines the way we divide up the bounding region
        RPlist = np.array(RPfaclist)*R
        print('RPlist', RPlist)
        if flag:
            outR.write(str(R)+'\n'); outR.flush()
        
        try:
            if (not feedforward) or (Lags is None):
                if not (Lags is None):
                     Minitklim = max(Minitklim,klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                     print('new Minitklim is', Minitklim)
                Lags, dualval, objval, klim = get_planewave_spherical_multiRegion_Psca_numpy(k,RPlist, chi, incPIm=incPIm, normtol=normtol, mpdps=mpdps, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, check_iter_period=check_iter_period)
            else: #feed previous optimal parameters as starting point for next optimization
                Minitklim = max(Minitklim,klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                print('new Minitklim is', Minitklim)
                if Minitklim > mp.dps: #just in case we need higher precision with growing polynomial order
                    mp.dps = Minitklim
                Lags, dualval, objval, klim = get_planewave_spherical_multiRegion_Psca_numpy(k,RPlist, chi, initLags=Lags, incPIm=incPIm, normtol=normtol, mpdps=mpdps, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, check_iter_period=check_iter_period)
        except (KeyboardInterrupt, SystemExit): #in case I'm testing and interrupting by hand
            raise
        
        duallist.append(dualval); objlist.append(objval); currentRlist.append(R)
            
        if flag:
            outdual.write(str(dualval)+'\n')
            outobj.write(str(objval)+'\n')

            np.save(filename+'_Rlist.npy', np.array(currentRlist)) #save as npy after each data point is calculated for easy plotting on-the-fly     
            np.save(filename+'_duallist.npy', np.array(duallist))
            np.save(filename+'_objlist.npy', np.array(objlist))
            outdual.flush()
            outobj.flush()
            outasym.flush()

    if flag:
        outR.close()
        outdual.close()
        outobj.close()

    return currentRlist, duallist, objlist
