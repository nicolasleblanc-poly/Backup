#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 30 11:21:41 2020

@author: pengning
"""

import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import mpmath
from mpmath import mp

from dualbound.Arnoldi.spherical_multiRegion_Green_Arnoldi import spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge, spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge
from dualbound.Lagrangian.spatialProjopt_Zops_numpy import get_ZTT_mineig, get_inc_ZTT_mineig
from dualbound.Lagrangian.spatialProjopt_feasiblept import spatialProjopt_find_feasiblept
from dualbound.Lagrangian.spatialProjopt_dualgradHess_fullSvec_numpy import get_inc_spatialProj_dualgradHess_fullS
from dualbound.Optimization.modSource_opt import modS_opt

def get_planewave_screening_datalists(k, Ro, Rd_Plist, chi, Eamp=1.0, mpdps=60, normtol = 1e-8, 
                                        gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, 
                                        Taylor_tol=1e-12, Unormtol=1e-6, includeI=True):
    """
    generate all relevant physics for the screening problem: minimize the field norm
    in a central spherical observation region for an incoming planewave impinging on a structure
    situated within an outer surrounding shell.
    Ro is the radius of the spherical observation region
    Rd_Plist stores the partitions for the design region
    Ro<=Rd_Plist[0]<Rd_Plist[-1]
    Eamp is the amplitude of the planewave field
    """
    
    if Ro>Rd_Plist[0]:
        raise ValueError('ordering of radii incorrect.')
    
    mp.dps = mpdps
    spherenorm = 4*np.pi * Rd_Plist[-1]**3 / 3 #norm of the unity amplitude planewave source over the entire enclosing sphere of both design and observation regions
    sumnorm = 0.0
    invchi = 1.0/chi
    
    #Rall_list contains all relevant radii of the problem is a merging of Ro1 and Ro2 into RPlist, so that we can generate the Green's function for the whole enclosing region using the Arnoldi process
    if Rd_Plist[0]-Ro > Rd_Plist[0]*0.01:
        Rall_list = np.concatenate(([Ro], Rd_Plist))
        print('shell between end of observation region and beginning of design region')
    else:
        Rall_list = Rd_Plist.copy()
        print('treating observation region to start effectively where the design region ends')
    print('Rd_Plist', Rd_Plist)
    print('Rall_list', Rall_list)
    
    Silist = []
    Salist = []
    
    Gmat_od_list = []
    Po_list = []
    Olist = []
    eps_o = 1e-2 #for computing bounds on the norm difference O = G_od.H @ G_od + eps_o*Proj_o, the final term is to numerically guarantee that ZTT is PD when all multipliers are 0 and should not influence final bound result
    
    Plistlist = []
    UPlistlist = [] #lists for the source field, target field Umatrix representation and U*Proj_i matrix representations to be constructed
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
        
        
        #calculate the RgM, RgN component coefficients of the planewave and distribute it over the regional subbases
        RgNe_coef = Eamp * 2 * (1j)**((l+1)%4) * np.sqrt((2*l+1)*np.pi/2) #the factor of 1/2 in the sqrt is the norm of the odd/even vector spherical harmonics
        RgMo_coef = RgNe_coef * 1j
        
        #construct RgM source vector
        M_Sivec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistM)-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            M_Sivec[subbasis_indlistM[i]] = np.complex(RgMo_coef*RgMnormlist[i])
        Silist.append(M_Sivec)
        
        #construct RgN source vector
        N_Sivec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistN)-1):
            N_Sivec[subbasis_indlistN[i]] = np.complex(RgNe_coef*RgNnormlist[i])
        Silist.append(N_Sivec)
        
        #for screening problem, target wave is 0
        Salist.append(np.zeros_like(M_Sivec))
        Salist.append(np.zeros_like(N_Sivec))
        
        
        sosqr = np.real(np.vdot(M_Sivec,M_Sivec)+np.vdot(N_Sivec,N_Sivec))
        sumnorm += sosqr
        print('sumnorm', sumnorm, 'spherenorm', spherenorm)
        
        if l==1:
            klim_update = 5 + 2*max(subbasis_indlistM[1]-subbasis_indlistM[0], subbasis_indlistN[1]-subbasis_indlistN[0])
        
        l += 1
    
    return Silist, Salist, Gmat_od_list, Olist, Po_list, Plistlist, UPlistlist, klim_update


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


def get_planewave_spherical_screening_bound(k, Ro, Rd_Plist, chi, Eamp=1.0, initLags=None, incPIm=False, incRegions=None, mpdps=60, normtol = 1e-8, 
                                                  gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, 
                                                  Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    
    Silist, Salist, Gmat_od_list, Olist, Po_list, Plistlist, UPlistlist, klim_update = get_planewave_screening_datalists(k, Ro, Rd_Plist, chi, Eamp=Eamp, mpdps=mpdps, normtol=normtol, 
                                                                                                                         gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, 
                                                                                                                         Taylor_tol=Taylor_tol, Unormtol=Unormtol, includeI=True)
    
    modenum = len(Silist)
    Lagnum = 2 + 2*(len(Rd_Plist)-1)
    print('Lagnum', Lagnum)
    print('len UPlistlist', len(UPlistlist[0]))
    
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


def sweep_norm_remainder_spherical_planewave_screening_varyR(k, Ro, Rd2list, num_Rd_div, chi, initLags=None, incPIm=False, incRegions=None, mpdps=60, normtol=1e-8, 
                                                                        gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, 
                                                                        Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, modSratio=1e-2, check_iter_period=20, filename=None, feedforward=True):
    
    flag = False
    if not (filename is None):
        flag = True
        outRd2 = open(filename+'_Rd2list.txt','w')
        outdual = open(filename+'_duallist.txt','w')
        outobj = open(filename+'_objlist.txt','w')
        outSinormsqr = open(filename+'_Sinormsqrlist.txt','w')
        outratio = open(filename+'_normsqrratiolist.txt', 'w')
        
    duallist = []; objlist = []
    currentRd2list = []
    Sinormsqrlist = []
    normsqrratiolist = []
    Lags = None
    klim = Minitklim
    
    for i in range(len(Rd2list)):
        Rd2 = Rd2list[i]
        Rd_Plist = np.linspace(Ro,Rd2, num_Rd_div+1)
        if flag:
            outRd2.write(str(Rd2)+'\n'); outRd2.flush()
        
        try:
            if (not feedforward) or (Lags is None):
                if not (Lags is None):
                    Minitklim = max(Minitklim,klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                    print('new Minitklim is', Minitklim)
                Lags, dualval, objval, Sinormsqr, klim = get_planewave_spherical_screening_bound(k, Ro, Rd_Plist, chi, initLags=None, incPIm=incPIm, incRegions=incRegions, mpdps=mpdps, normtol = normtol, 
                                                                                                       gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, 
                                                                                                       Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
            else: #feed previous optimal parameters as starting point for next optimization
                Minitklim = max(Minitklim,klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                print('new Minitklim is', Minitklim)
                if Minitklim > mp.dps: #just in case we need higher precision with growing polynomial order
                    mp.dps = Minitklim
                Lags, dualval, objval, Sinormsqr, klim = get_planewave_spherical_screening_bound(k, Ro, Rd_Plist, chi, initLags=Lags, incPIm=incPIm, incRegions=incRegions, mpdps=mpdps, normtol = normtol, 
                                                                                                       gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, 
                                                                                                       Taylor_tol=Taylor_tol, Unormtol=Unormtol, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
        except (KeyboardInterrupt, SystemExit): #in case I'm testing and interrupting by hand
            raise
        
        duallist.append(dualval); objlist.append(objval)
        currentRd2list.append(Rd2)
        Sinormsqrlist.append(Sinormsqr)
        normsqrratio = dualval / Sinormsqr
        normsqrratiolist.append(normsqrratio)
        
        if flag:
            outdual.write(str(dualval)+'\n'); outdual.flush()
            outobj.write(str(objval)+'\n'); outobj.flush()
            outSinormsqr.write(str(Sinormsqr)+'\n'); outSinormsqr.flush()
            outratio.write(str(normsqrratio)+'\n'); outratio.flush()
            
            np.save(filename+'_Rd2list.npy', np.array(currentRd2list))
            np.save(filename+'_duallist.npy', np.array(duallist))
            np.save(filename+'_objlist.npy', np.array(objlist))
            np.save(filename+'_Sinormsqrlist.npy', np.array(Sinormsqrlist))
            np.save(filename+'_normsqrratiolist.npy', np.array(normsqrratiolist))
    
    if flag:
        outdual.close()
        outobj.close()
        outSinormsqr.close()
        outratio.close()
    
    return currentRd2list, duallist, objlist, Sinormsqrlist, normsqrratiolist