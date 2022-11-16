#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 17:32:44 2020

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
from dualbound.Arnoldi.spherical_multiRegion_Green_Arnoldi import spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge, spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge
from dualbound.Lagrangian.spatialProjopt_Zops_numpy import get_ZTT_mineig, get_inc_ZTT_mineig
from dualbound.Lagrangian.spatialProjopt_feasiblept import spatialProjopt_find_feasiblept
from dualbound.Lagrangian.spatialProjopt_dualgradHess_fullSvec_numpy import get_inc_spatialProj_dualgradHess_fullS
from dualbound.Optimization.modSource_opt import modS_opt
from eqconstraint_xdipole_spherical_domain import get_real_RgNM_l_coeffs_for_xdipole_field


def get_xdipole_sphere_multiRegion_S1list_Ulists_Projlists_numpy(k,RPlist,dist,chi,
                                                                 S1list, Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                 mpdps=60, normtol = 1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True):
    
    #along with the old xdipole data, this method generates the block diagonals of spatial shell projection operators
    #RPlist stores the boundaries between the different projection regions; RPlist[-1] is the radius of the entire bounding sphere
    #length of Ulist is the number of different modes involved, for xdipoles there are RgM_l1o and RgN_l1e waves
    #Plistlist and UPlistlist are organized such that the first index is mode, and second index is projection operator number
    #subbasis_ind_listlist is a list for all the mode's lists of the starting indices of each projection region's subbasis
    #if includeI==True, then we include the original constraints with P0=I
    sumnorm = 0
    mp.dps=mpdps
    #we are going to modify the list arguments in place
    l=1
    invchi = 1.0/chi
    while True:
        print('at mode number', l)
        if 2*l>len(Plistlist): #extend the Plistlists and UPlistlists if necessary, factor of 2 to account for both M and N waves
            if l==1:
                Mklim = Minitklim; Nklim = Ninitklim
            else:
                #the first subregion is the inner sphere; its Arnoldi process uses the old Taylor Arnoldi code
                Mklim = 2*(subbasis_indlistlist[-2][1]-subbasis_indlistlist[-2][0])+5
                Nklim = 2*(subbasis_indlistlist[-1][1]-subbasis_indlistlist[-1][0])+5
            print(Nklim)
            
            #################first do the M wave###################################################
            Gmat, Uconj, RgMnormlist, subbasis_indlist, fullrgrid, All_fullr_unitMvecs = spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge(l,k,RPlist, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Mklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
            U = Uconj.conjugate()
            
            if includeI: #now extend Plistlist and UPlistlist
                Plist = [np.eye(U.shape[0])]
                UPlist = [U]
            else:
                Plist = []
                UPlist = []
                
            for i in range(len(RPlist)):
                Proj = np.zeros_like(U)
                subbasis_head = subbasis_indlist[i]; subbasis_tail = subbasis_indlist[i+1]
                Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
                UProj = U @ Proj
                Plist.append(Proj)
                UPlist.append(UProj)
        
            Plistlist.append(Plist)
            UPlistlist.append(UPlist)
            Rgnormlistlist.append(RgMnormlist)
            subbasis_indlistlist.append(subbasis_indlist)
            
            #the do N wave
            Gmat, Uconj, RgNnormlist, subbasis_indlist, fullrgrid, All_fullr_unitBvecs,All_fullr_unitPvecs = spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge(l,k,RPlist, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Nklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
            U = Uconj.conjugate() #the U generated in Green_Taylor_Arnoldi is an older definition that corresponds to U^\dagger in our current notation
                
            if includeI: #now extend Plistlist and UPlistlist
                Plist = [np.eye(U.shape[0])]
                UPlist = [U]
            else:
                Plist = []
                UPlist = []
                
            for i in range(len(RPlist)):
                Proj = np.zeros_like(U)
                subbasis_head = subbasis_indlist[i]; subbasis_tail = subbasis_indlist[i+1]
                Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
                UProj = U @ Proj
                Plist.append(Proj)
                UPlist.append(UProj)
        
            Plistlist.append(Plist)
            UPlistlist.append(UPlist)
            Rgnormlistlist.append(RgNnormlist)
            subbasis_indlistlist.append(subbasis_indlist)
            
        
        np.seterr(under='warn')
        #the coeffs of the source in terms of the non-normalized regular waves with real azimuthal angle dependence
        coeffs = get_real_RgNM_l_coeffs_for_xdipole_field(l,k,RPlist[-1]+dist, wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10)
        #the coefficients in coeffs are stored in the order of [RgM_l,1,e RgM_l,1,o RgN_l,1,e RgN_l,1,o]
            
        #construct the M source vector
        M_S1vec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-2])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            M_S1vec[subbasis_indlistlist[2*l-2][i]] = np.complex(coeffs[1]*Rgnormlistlist[2*l-2][i])/np.sqrt(2) #factor of 1/sqrt(2) comes from normalizing the real azimuthal vector spherical harmonics
        S1list.append(M_S1vec)

        #construct the N source vector
        N_S1vec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-1])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            N_S1vec[subbasis_indlistlist[2*l-1][i]] = np.complex(coeffs[2]*Rgnormlistlist[2*l-1][i])/np.sqrt(2)
        S1list.append(N_S1vec)
        
        sosqr = np.real(np.vdot(M_S1vec,M_S1vec)+np.vdot(N_S1vec,N_S1vec))
        sumnorm += sosqr #check with old code and see the sumnorms are the same
        print('sumnorm,', sumnorm)
        if l>2 and sosqr<sumnorm*normtol:
            break
        l+=1
        

def get_ext_Prad_ZTS_Slist(Lags, S1list, Plistlist):
    ZTS_Slist = []
    for mode in range(len(S1list)):
        Plist = Plistlist[mode]
        ZTS = np.zeros_like(Plist[0], dtype=np.complex)
        for i in range(len(Plist)):
            ZTS += (Lags[2*i]+1j*Lags[2*i+1])*Plist[i].conj().T/2
            
        ZTS_S = ZTS @ S1list[mode] + (1j/2)*np.conj(S1list[mode]) #for external dipoles, S2 = S1*
        ZTS_Slist.append(ZTS_S)
    return ZTS_Slist

def get_ext_Prad_gradZTS_Slist(Lags, S1list, Plistlist):
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


def get_xdipole_spherical_multiRegion_shellProj_Psca_numpy(k,RPlist,dist,chi, S1list, Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                           initLags=None, incPIm=False, incRegions=None, mpdps=60, maxveclim=40, normtol=1e-4, gridpts=1000, Minitklim=25, Ninitklim=25, Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, modSratio=1e-2, check_iter_period=20):

    mp.dps = mpdps
    get_xdipole_sphere_multiRegion_S1list_Ulists_Projlists_numpy(k,RPlist,dist,chi,
                                                                 S1list, Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                 mpdps=mpdps, normtol = normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol, includeI=True)
    
    
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
    
    ZTS_Slistfunc = lambda L, Slist: get_ext_Prad_ZTS_Slist(L, Slist, Plistlist)
    gradZTS_Slistfunc = lambda L, Slist: get_ext_Prad_gradZTS_Slist(L, Slist, Plistlist)
    dgHfunc = lambda dof, dofgrad, dofHess, Slist, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fullS(dof, dofgrad, dofHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc, get_grad=get_grad, get_Hess=get_Hess)
    mineigfunc = lambda dof, eigvals_only=False: get_inc_ZTT_mineig(dof, include, Olist, UPlistlist, eigvals_only=eigvals_only)

    opt_incLags, opt_incgrad, opt_dual, opt_obj = modS_opt(Lags[include], S1list, dgHfunc, mineigfunc, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
    optLags = np.zeros(Lagnum)
    optgrad = np.zeros(Lagnum)
    optLags[include] = opt_incLags
    optgrad[include] = opt_incgrad
    print('the remaining constraint violations')
    print(optgrad)
    #prefactors from physics
    Z=1
    return optLags, k*opt_dual/2/Z, k*opt_obj/2/Z, 2*(subbasis_indlistlist[0][1]-subbasis_indlistlist[0][0])+5 #final returned value is useful for certain sweeps to determine polynomial order for the Arnoldi process



def sweep_Psca_xdipole_multipleRegion_shellProj_varydist_np(k, RPlist, distlist, chi,
                                                            incPIm=False, incRegions=None, normtol = 1e-4, mpdps=60, maxveclim=40, gridpts=1000, Minitklim=25, Ninitklim=25, Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, check_iter_period=20,
                                                            filename=None, feedforward=True):
    #for now let the external calling script set the domains
    flag = False
    if not (filename is None):
        flag = True
        outdist = open(filename+'_distlist.txt','w'); outdual = open(filename+'_duallist.txt','w')
        outobj = open(filename+'_objlist.txt','w'); outnPrad = open(filename+'_nPradlist.txt','w')
    
    wig_1llp1_000=[]; wig_1llm1_000=[]; wig_1llp1_1m10=[]; wig_1llm1_1m10=[]
    Plistlist=[]
    UPlistlist=[]
    Rgnormlistlist=[]
    subbasis_indlistlist=[]
    #these lists will be gradually filled up as sweep progresses
    #since we fix R, the U matrices and Proj matrices only need to be computed once
    
    duallist=[]; objlist=[]; nPradlist=[]; currentdistlist=[]
    dRad = 1.0/(12*np.pi)  #vacuum dipole radiation contribution, for calculating normalized Prad
    Lags = None
    klim=Minitklim
    for i in range(len(distlist)):
        dist = distlist[i]

        if flag:
            outdist.write(str(dist)+'\n')
            outdist.flush()
            
        try:
            if (not feedforward) or (Lags is None):
                if not (Lags is None):
                    Minitklim = max(Minitklim, klim+3); Ninitklim = max(Ninitklim,klim+3) #change Ninitklim based on required size from previous iteration
                    print('new Minitklim is', Minitklim, 'new Ninitklim is', Ninitklim)
                Lags, dualval, objval, klim = get_xdipole_spherical_multiRegion_shellProj_Psca_numpy(k,RPlist,dist,chi, [], Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                                                     initLags=None, incPIm=incPIm, incRegions=incRegions, mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, check_iter_period=check_iter_period)
                #we may lose strong duality so never start from rand init
            else: #feed previous optimal parameters as starting point for next optimization
                Minitklim = max(Minitklim, klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                print('new Minitklim is', Minitklim, 'new Ninitklim is', Ninitklim)
                if Minitklim > mp.dps:
                    mp.dps = Minitklim
                Lags, dualval, objval, klim = get_xdipole_spherical_multiRegion_shellProj_Psca_numpy(k,RPlist,dist,chi, [], Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                                                     initLags=Lags, incPIm=incPIm, incRegions=incRegions, mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, check_iter_period=check_iter_period)
                #we may lose strong duality so never start from rand init
        except (KeyboardInterrupt, SystemExit): #in case I am interrupting by hand
            raise

        nPrad = (dualval+dRad)/dRad
        duallist.append(dualval); objlist.append(objval);  
        currentdistlist.append(dist)
        nPradlist.append(nPrad)
            
        if flag:
            outdual.write(str(dualval)+'\n')
            outobj.write(str(objval)+'\n')
            outnPrad.write(str(nPrad)+'\n')

            np.save(filename+'_distlist.npy', np.array(currentdistlist)) #save as npy after each data point is calculated for easy plotting on-the-fly     
            np.save(filename+'_duallist.npy', np.array(duallist))
            np.save(filename+'_objlist.npy', np.array(objlist))
            np.save(filename+'_nPradlist.npy', np.array(nPradlist))
            outdual.flush()
            outobj.flush()
            outnPrad.flush()
        print(len(UPlistlist), len(wig_1llp1_000), len(wig_1llm1_000))

    if flag:
        outdist.close()
        outdual.close()
        outobj.close()
        outnPrad.close()
        
    return currentdistlist, duallist, objlist, nPradlist