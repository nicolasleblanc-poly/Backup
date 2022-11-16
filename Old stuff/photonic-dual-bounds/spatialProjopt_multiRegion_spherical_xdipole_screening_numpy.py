#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  2 21:32:55 2020

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

def get_xdipole_sphere_screening_datalists_numpy(k, dist, Ro, Rd_Plist, chi,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 mpdps=60, normtol = 1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True):

    """
    computes the physics datalists needed for computing bounds on screening of an xdipole field
    with a spherical shell domain from a center sphere observation area
    Plistlist, UPlistlist, ... are sent in as parameters so we can store them external of this method,
    which in the case where we sweep dipole distance from shell and the shell geometry is fixed allows
    us to avoid recomputing these quantities between different dipole distance inputs
    """
    if Ro>Rd_Plist[0]:
        raise ValueError('ordering of radii incorrect.')
    
    #Rall_list contains all relevant radii of the problem is a merging of Ro1 and Ro2 into RPlist, so that we can generate the Green's function for the whole enclosing region using the Arnoldi process
    if Rd_Plist[0]-Ro > Rd_Plist[0]*0.01:
        Rall_list = np.concatenate(([Ro], Rd_Plist))
        print('shell between end of observation region and beginning of design region')
    else:
        Rall_list = Rd_Plist.copy()
        print('treating observation region to start effectively where the design region ends')
    print('Rd_Plist', Rd_Plist)
    print('Rall_list', Rall_list)
    
    eps_o = 1e-2 #for computing bounds on the norm difference O = G_od.H @ G_od + eps_o*Proj_o, the final term is to numerically guarantee that ZTT is PD when all multipliers are 0 and should not influence final bound result
    
    sumnorm = 0
    mp.dps=mpdps
    #we are going to modify the list arguments in place
    l=1
    invchi = 1.0/chi

    while True:
        print('at mode number', l)
        if 2*l>len(Plistlist): #extend the datalists if necessary, factor of 2 to account for both M and N waves
            if l==1:
                Mklim = Minitklim; Nklim = Ninitklim
            else:
                #the first subregion is the inner sphere; its Arnoldi process uses the old Taylor Arnoldi code
                Mklim = 2*(subbasis_indlistlist[-2][1]-subbasis_indlistlist[-2][0])+5
                Nklim = 2*(subbasis_indlistlist[-1][1]-subbasis_indlistlist[-1][0])+5
                
            ############################M wave component##############################
            Gmat, Uconj, RgMnormlist, subbasis_indlistM, fullrgrid, All_fullr_unitMvecs = spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge(l,k,Rall_list, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Mklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
            U = Uconj.conjugate()
            
            Proj_o = np.zeros_like(U)
            observe_basis_head = 0
            observe_basis_tail = subbasis_indlistM[1]
            Proj_o[observe_basis_head:observe_basis_tail,observe_basis_head:observe_basis_tail] = np.eye(observe_basis_tail-observe_basis_head)
            
            Proj_d = np.zeros_like(U)
            design_basis_head = subbasis_indlistM[len(Rall_list)-len(Rd_Plist)+1]
            design_basis_tail = subbasis_indlistM[-1]
            Proj_d[design_basis_head:design_basis_tail,design_basis_head:design_basis_tail] = np.eye(design_basis_tail-design_basis_head)
        
            U = Proj_d @ U @ Proj_d #restrict actual U matrix to be within the design region
            Gmat_od = Proj_o @ Gmat @ Proj_d
            Gmat_od_list.append(Gmat_od)
            Po_list.append(Proj_o)
            Olist.append(Gmat_od.T.conj() @ Gmat_od + eps_o*(np.eye(subbasis_indlistM[-1])-Proj_d))
            
            if includeI:
                Plist = [Proj_d]
                UPlist = [U]
            else:
                Plist = []
                UPlist = []
            
            for i in range(len(Rd_Plist)-1): #add all of the projection subregions for the design domain
                #the -1 is because the innermost sphere is not part of the design region
                Proj = np.zeros_like(U)
                subbasis_ind = len(Rall_list)-len(Rd_Plist)+1 + i
                subbasis_head = subbasis_indlistM[subbasis_ind]
                subbasis_tail = subbasis_indlistM[subbasis_ind+1]
                Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
                UProj = U @ Proj
                Plist.append(Proj)
                UPlist.append(UProj)
                
            Plistlist.append(Plist)
            UPlistlist.append(UPlist)
            Rgnormlistlist.append(RgMnormlist)
            subbasis_indlistlist.append(subbasis_indlistM)
            
            ############################N wave component##############################
            Gmat, Uconj, RgNnormlist, subbasis_indlistN, fullrgrid, All_fullr_unitBvecs, All_fullr_unitPvecs = spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge(l,k,Rall_list, invchi, gridpts=gridpts, mpdps=mpdps, maxveclim=maxveclim, klim=Nklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
            U = Uconj.conjugate()
            
            Proj_o = np.zeros_like(U)
            observe_basis_head = 0
            observe_basis_tail = subbasis_indlistN[1]
            Proj_o[observe_basis_head:observe_basis_tail,observe_basis_head:observe_basis_tail] = np.eye(observe_basis_tail-observe_basis_head)
            
            Proj_d = np.zeros_like(U)
            design_basis_head = subbasis_indlistN[len(Rall_list)-len(Rd_Plist)+1]
            design_basis_tail = subbasis_indlistN[-1]
            Proj_d[design_basis_head:design_basis_tail,design_basis_head:design_basis_tail] = np.eye(design_basis_tail-design_basis_head)
            
            U = Proj_d @ U @ Proj_d #restrict actual U matrix to be within the design region
            Gmat_od = Proj_o @ Gmat @ Proj_d
            Gmat_od_list.append(Gmat_od)
            Po_list.append(Proj_o)
            Olist.append(Gmat_od.T.conj() @ Gmat_od + eps_o*(np.eye(subbasis_indlistN[-1])-Proj_d))
            
            if includeI:
                Plist = [Proj_d]
                UPlist = [U]
            else:
                Plist = []
                UPlist = []
                
            for i in range(len(Rd_Plist)-1): #add all of the projection subregions for the design domain
                Proj = np.zeros_like(U)
                subbasis_ind = len(Rall_list)-len(Rd_Plist)+1 + i
                subbasis_head = subbasis_indlistN[subbasis_ind]
                subbasis_tail = subbasis_indlistN[subbasis_ind+1]
                Proj[subbasis_head:subbasis_tail,subbasis_head:subbasis_tail] = np.eye(subbasis_tail-subbasis_head)
                UProj = U @ Proj
                Plist.append(Proj)
                UPlist.append(UProj)
                
            Plistlist.append(Plist)
            UPlistlist.append(UPlist)
            Rgnormlistlist.append(RgNnormlist)
            subbasis_indlistlist.append(subbasis_indlistN)
            
        ###############calculate the coefficients for Si, the xdipole wave###############
        np.seterr(under='warn')
        #the coeffs of the source in terms of the non-normalized regular waves with real azimuthal angle dependence
        coeffs = get_real_RgNM_l_coeffs_for_xdipole_field(l,k,Rall_list[-1]+dist, wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10)
        #the coefficients in coeffs are stored in the order of [RgM_l,1,e RgM_l,1,o RgN_l,1,e RgN_l,1,o]
            
        #construct the M source vector
        M_Sivec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-2])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            M_Sivec[subbasis_indlistlist[2*l-2][i]] = np.complex(coeffs[1]*Rgnormlistlist[2*l-2][i])/np.sqrt(2) #factor of 1/sqrt(2) comes from normalizing the real azimuthal vector spherical harmonics
        Silist.append(M_Sivec)

        #construct the N source vector
        N_Sivec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-1])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            N_Sivec[subbasis_indlistlist[2*l-1][i]] = np.complex(coeffs[2]*Rgnormlistlist[2*l-1][i])/np.sqrt(2)
        Silist.append(N_Sivec)
        
        #for screening problem, target wave is 0
        Salist.append(np.zeros_like(M_Sivec))
        Salist.append(np.zeros_like(N_Sivec))
        
        sosqr = np.real(np.vdot(M_Sivec,M_Sivec)+np.vdot(N_Sivec,N_Sivec))
        sumnorm += sosqr #check with old code and see the sumnorms are the same
        print('sumnorm,', sumnorm)
        if l>2 and sosqr<sumnorm*normtol:
            break
        
        l+=1
    
    
    return 5 + 2*max(subbasis_indlistlist[0][1]-subbasis_indlistlist[0][0], subbasis_indlistlist[1][1]-subbasis_indlistlist[1][0])
        

def get_comm_normdiffsqr_ZTS_Slist(Lags, Silist, Salist, Plistlist, Gmat_od_list):
    ZTS_Slist = []
    for mode in range(len(Silist)):
        Plist = Plistlist[mode]
        ZTS = np.zeros_like(Plist[0], dtype=np.complex)
        for i in range(len(Plist)):
            ZTS += (Lags[2*i] + 1j*Lags[2*i+1])*Plist[i].conj().T/2
        
        ZTS_S = ZTS @ Silist[mode] + Gmat_od_list[mode].conj().T @ (Salist[mode]-Silist[mode])
        ZTS_Slist.append(ZTS_S)
    return ZTS_Slist

    
def get_comm_normdiffsqr_gradZTS_Slist(Lags, Silist, Salist, Plistlist):
    gradZTS_Slist = []
    for mode in range(len(Silist)):
        Plist = Plistlist[mode]
        gradZTS_S = []
        for i in range(len(Plist)):
            P_S = Plist[i] @ Silist[mode]
            gradZTS_S.append(P_S/2)
            gradZTS_S.append(1j*P_S/2)
        gradZTS_Slist.append(gradZTS_S)
    return gradZTS_Slist


def get_xdipole_screening_bound(k, dist, Ro, Rd_Plist, chi,
                                Silist, Salist, Olist, Gmat_od_list, Po_list,
                                Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                initLags=None, incPIm=False, incRegions=None, 
                                mpdps=60, normtol = 1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6,
                                opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    
    mp.dps=mpdps
    klim_update = get_xdipole_sphere_screening_datalists_numpy(k, dist, Ro, Rd_Plist, chi,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
    
    modenum = len(Silist)
    Lagnum = 2 + 2*(len(Rd_Plist)-1)
    
    include=[True]*Lagnum
    if incRegions is None: #use default region inclusion: every region except the innermost
        include[2] = False; include[3] = False
    else:
        for i in range(len(incRegions)):
            include[2+2*i] = incRegions[i]; include[3+2*i] = incRegions[i]
    
    if not incPIm:
        for i in range(1,len(Rd_Plist)-1+1): #i starts from 1 because we always want to keep the original Im constraint
            include[2*i+1] = False
    
    if initLags is None:
        Lags = spatialProjopt_find_feasiblept(Lagnum, include, Olist, UPlistlist)
    else:
        Lags = initLags.copy()
        Lags[1] = np.abs(Lags[1])+0.01
    
    ########################constant part of the optimization objective###################
    normsqr_const = 0.0 #constant part that upon summation gives || Proj_o @ (|E_t>-|E_a>) ||^2
    normsqr_Po_Si = 0.0
    for mode in range(modenum):
        normsqr_const += -np.real(np.vdot(Silist[mode], Po_list[mode] @ Silist[mode])) - np.real(np.vdot(Salist[mode], Po_list[mode] @ Salist[mode])) + 2*np.real(np.vdot(Salist[mode], Po_list[mode] @ Silist[mode]))
        normsqr_Po_Si += np.real(np.vdot(Po_list[mode] @ Silist[mode],Po_list[mode] @ Silist[mode]))
    print('constant part of || Proj_o @ (|E_t>-|E_a>) ||^2', normsqr_const, 'Si normsqr over observation region', normsqr_Po_Si, flush=True)
    
    
    validLagsfunc = lambda L: get_ZTT_mineig(L, Olist, UPlistlist, eigvals_only=True)[1] #[1] because we only need the minimum eigenvalue part of the returned values
    while validLagsfunc(Lags)<0:
        print('zeta', Lags[1])
        Lags[1] *= 1.5 #enlarge zeta until we enter domain of validity
    print('initial Lags', Lags)
    
    ZTS_Slistfunc = lambda L, Slist: get_comm_normdiffsqr_ZTS_Slist(L, Slist, Salist, Plistlist, Gmat_od_list)
    gradZTS_Slistfunc = lambda L, Slist: get_comm_normdiffsqr_gradZTS_Slist(L, Slist, Salist, Plistlist)
    
    dgHfunc = lambda dof, dofgrad, dofHess, Slist, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fullS(dof, dofgrad, dofHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc,dualconst=normsqr_const, get_grad=get_grad, get_Hess=get_Hess)
    mineigfunc = lambda dof, eigvals_only=False: get_inc_ZTT_mineig(dof, include, Olist, UPlistlist, eigvals_only=eigvals_only)
    
    opt_incLags, opt_incgrad, opt_dual, opt_obj = modS_opt(Lags[include], Silist, dgHfunc, mineigfunc, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
    optLags = np.zeros(Lagnum)
    optgrad = np.zeros(Lagnum)
    optLags[include] = opt_incLags
    optgrad[include] = opt_incgrad
    print('the remaining constraint violations')
    print(optgrad)
    
    return optLags, -opt_dual, -opt_obj, normsqr_Po_Si, klim_update


def sweep_norm_remainder_spherical_xdipole_screening_varydist(k, distlist, Ro, Rd_Plist, chi, incPIm=False, incRegions=None, normtol = 1e-4, mpdps=60, maxveclim=40, gridpts=1000, Minitklim=25, Ninitklim=25, Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, modSratio=1e-2, check_iter_period=20,
                                                            filename=None, feedforward=True):
    
    #for now let the external calling script set the domains
    flag = False
    if not (filename is None):
        flag = True
        outdist = open(filename+'_distlist.txt','w')
        outdual = open(filename+'_duallist.txt','w')
        outobj = open(filename+'_objlist.txt','w')
        outSinormsqr = open(filename+'_Sinormsqrlist.txt','w')
        outratio = open(filename+'_normsqrratiolist.txt', 'w')
    
    wig_1llp1_000=[]; wig_1llm1_000=[]; wig_1llp1_1m10=[]; wig_1llm1_1m10=[]
    Olist=[]
    Gmat_od_list=[]
    Po_list=[]
    Plistlist=[]
    UPlistlist=[]
    Rgnormlistlist=[]
    subbasis_indlistlist=[]
    #these lists will be gradually filled up as sweep progresses
    #since we fix the geometry, the U matrices and Proj matrices only need to be computed once
    
    duallist = []
    objlist = []
    currentdistlist = []
    Sinormsqrlist = []
    normsqrratiolist = []
    Lags = None
    klim = Minitklim
    
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
                Lags, dualval, objval, Sinormsqr, klim = get_xdipole_screening_bound(k, dist, Ro, Rd_Plist, chi,
                                                                                     [], [], Olist, Gmat_od_list, Po_list,
                                                                                     Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                                                     wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                                     initLags=None, incPIm=False, incRegions=None, 
                                                                                     mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol,
                                                                                     opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
                #we may lose strong duality so never start from rand init
            else: #feed previous optimal parameters as starting point for next optimization
                Minitklim = max(Minitklim, klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                print('new Minitklim is', Minitklim, 'new Ninitklim is', Ninitklim)
                if Minitklim > mp.dps:
                    mp.dps = Minitklim
                Lags, dualval, objval, Sinormsqr, klim = get_xdipole_screening_bound(k, dist, Ro, Rd_Plist, chi,
                                                                                     [], [], Olist, Gmat_od_list, Po_list,
                                                                                     Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                                                     wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                                                     initLags=Lags, incPIm=False, incRegions=None, 
                                                                                     mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol,
                                                                                     opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
                #we may lose strong duality so never start from rand init
        except (KeyboardInterrupt, SystemExit): #in case I am interrupting by hand
            raise
        
        duallist.append(dualval)
        objlist.append(objval)
        currentdistlist.append(dist)
        Sinormsqrlist.append(Sinormsqr)
        normsqrratio = dualval / Sinormsqr
        normsqrratiolist.append(normsqrratio)
        
        if flag:
            outdual.write(str(dualval)+'\n'); outdual.flush()
            outobj.write(str(objval)+'\n'); outobj.flush()
            outSinormsqr.write(str(Sinormsqr)+'\n'); outSinormsqr.flush()
            outratio.write(str(normsqrratio)+'\n'); outratio.flush()
            
            np.save(filename+'_distlist.npy', np.array(currentdistlist))
            np.save(filename+'_duallist.npy', np.array(duallist))
            np.save(filename+'_objlist.npy', np.array(objlist))
            np.save(filename+'_Sinormsqrlist.npy', np.array(Sinormsqrlist))
            np.save(filename+'_normsqrratiolist.npy', np.array(normsqrratiolist))
    
    if flag:
        outdist.close()
        outdual.close()
        outobj.close()
        outSinormsqr.close()
        outratio.close()
    
    return currentdistlist, duallist, objlist, Sinormsqrlist, normsqrratiolist