#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 10:47:26 2020

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
import matplotlib.pyplot as plt
from dualbound.Arnoldi.spherical_multiRegion_Green_Arnoldi import spherical_multiRegion_Green_Arnoldi_Mmn_Uconverge, spherical_multiRegion_Green_Arnoldi_Nmn_Uconverge
from dualbound.Lagrangian.spatialProjopt_2source_Zops_numpy import get_ZTT_mineig, get_inc_ZTT_mineig
from dualbound.Lagrangian.spatialProjopt_2source_dualgradHess_fullSvec_numpy import get_inc_spatialProj_dualgradHess_fullS, get_2source_separate_duals, get_2source_separate_obj
from dualbound.Lagrangian.spatialProjopt_2source_feasiblept import spatialProjopt_2source_find_feasiblept
from dualbound.Optimization.modSource_opt import modS_opt
from eqconstraint_xdipole_spherical_domain import get_real_RgNM_l_coeffs_for_xdipole_field


def get_sphere_planewave_xdipole_screening_datalists_numpy(k, dist, Ro, Rd_Plist, chi, xdfac,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 mpdps=60, normtol=1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True):
    
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
    
    pl_spherenorm = 4*np.pi * Rd_Plist[-1]**3 / 3
    pl_sumnorm = 0
    xd_sumnorm = 0
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
            num_basis = U.shape[0]
            
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
            Omat = np.zeros((2*num_basis, 2*num_basis), dtype=np.complex)
            np.seterr(under='warn')
            try:
                Omat_block = Gmat_od.T.conj() @ Gmat_od + eps_o*(np.eye(subbasis_indlistM[-1])-Proj_d)
            except FloatingPointError:
                print('M','l',l)
                print(Gmat_od)
                raise
            Omat[:num_basis,:num_basis] = Omat_block[:,:]
            Omat[num_basis:,num_basis:] = Omat_block[:,:]
            #print('Omat for M')
            #plt.figure()
            #plt.imshow(np.abs(Omat))
            #plt.show()
            Olist.append(Omat)
            
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
            num_basis = U.shape[0]
            
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
            Omat = np.zeros((2*num_basis, 2*num_basis), dtype=np.complex)
            np.seterr(under='warn')
            try:
                Omat_block = Gmat_od.T.conj() @ Gmat_od + eps_o*(np.eye(subbasis_indlistN[-1])-Proj_d)
            except FloatingPointError:
                print('N','l',l)
                print(Gmat_od)
                raise
            Omat[:num_basis,:num_basis] = Omat_block[:,:]
            Omat[num_basis:,num_basis:] = Omat_block[:,:]
            Olist.append(Omat)
            
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
            
        #####calculate the coefficients for Si, including the planewave and the xdipole wave####
        np.seterr(under='warn')
        #the planewave
        RgNe_coef = 2 * (1j)**((l+1)%4) * np.sqrt((2*l+1)*np.pi/2) #the factor of 1/2 in the sqrt is the norm of the odd/even vector spherical harmonics
        RgMo_coef = RgNe_coef * 1j
        
        #construct RgM source vector
        pl_M_Sivec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-2])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            pl_M_Sivec[subbasis_indlistlist[2*l-2][i]] = np.complex(RgMo_coef*Rgnormlistlist[2*l-2][i])

        #construct RgN source vector
        pl_N_Sivec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-1])-1):
            pl_N_Sivec[subbasis_indlistlist[2*l-1][i]] = np.complex(RgNe_coef*Rgnormlistlist[2*l-1][i])

        #the coeffs of the source in terms of the non-normalized regular waves with real azimuthal angle dependence
        xd_coeffs = xdfac * np.array(get_real_RgNM_l_coeffs_for_xdipole_field(l,k,Rall_list[-1]+dist, wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10))
        #the coefficients in coeffs are stored in the order of [RgM_l,1,e RgM_l,1,o RgN_l,1,e RgN_l,1,o]
            
        #construct the xdipole M source vector
        xd_M_Sivec = np.zeros(UPlistlist[2*l-2][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-2])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            xd_M_Sivec[subbasis_indlistlist[2*l-2][i]] = np.complex(xd_coeffs[1]*Rgnormlistlist[2*l-2][i])/np.sqrt(2) #factor of 1/sqrt(2) comes from normalizing the real azimuthal vector spherical harmonics


        #construct the xdipole N source vector
        xd_N_Sivec = np.zeros(UPlistlist[2*l-1][0].shape[0], dtype=np.complex)
        for i in range(len(subbasis_indlistlist[2*l-1])-1): #-1 because last index of subbasis_indlist is total number of basis vectors
            xd_N_Sivec[subbasis_indlistlist[2*l-1][i]] = np.complex(xd_coeffs[2]*Rgnormlistlist[2*l-1][i])/np.sqrt(2)


        Silist.append(np.concatenate((pl_M_Sivec, xd_M_Sivec)))
        Silist.append(np.concatenate((pl_N_Sivec, xd_N_Sivec)))
        
        
        #for screening problem, target wave is 0
        Salist.append(np.zeros_like(Silist[-2]))
        Salist.append(np.zeros_like(Silist[-1]))
        
        pl_sosqr = np.real(np.vdot(pl_M_Sivec,pl_M_Sivec)+np.vdot(pl_N_Sivec,pl_N_Sivec))
        pl_sumnorm += pl_sosqr
        print('planewave sumnorm', pl_sumnorm, 'planewave spherenorm', pl_spherenorm)
        
        xd_sosqr = np.real(np.vdot(xd_M_Sivec,xd_M_Sivec)+np.vdot(xd_N_Sivec,xd_N_Sivec))
        xd_sumnorm += xd_sosqr
        print('xd sumnorm,', xd_sumnorm)
        if l>2 and xd_sosqr<xd_sumnorm*normtol and pl_spherenorm-pl_sumnorm<normtol*pl_spherenorm:
            #break when both the planewave and xdipole norm summation tolerances have been met
            break
        
        l+=1
    
    return 5 + 2*max(subbasis_indlistlist[0][1]-subbasis_indlistlist[0][0], subbasis_indlistlist[1][1]-subbasis_indlistlist[1][0])


def get_2source_comm_normdiffsqr_ZTS_Slist(Lags, Silist, Salist, Plistlist, Gmat_od_list):
    """
    the matrices in Plistlist and Gmat_od_list have dimensions the same as the # of spatial basis vectors
    Silist and Salist store vectors with length twice that of # of spatial basis vectors
    Si/Sa breaks down into [Si1; Si2] and [Sa1; Sa2]
    """
    
    ZTS_Slist = []
    for mode in range(len(Silist)):
        Plist = Plistlist[mode]
        num_basis = Plist[0].shape[0]
        shape_ZTS = (2*num_basis,2*num_basis)
        ZTS = np.zeros(shape_ZTS, dtype=np.complex)
        for i in range(len(Plist)):
            PH = Plist[i].T.conj()
            alpha_re11 = Lags[8*i]; alpha_im11 = Lags[8*i+4]
            alpha_re12 = Lags[8*i+1]; alpha_im12 = Lags[8*i+5]
            alpha_re21 = Lags[8*i+2]; alpha_im21 = Lags[8*i+6]
            alpha_re22 = Lags[8*i+3]; alpha_im22 = Lags[8*i+7]
            ZT1S1 = (alpha_re11 + 1j*alpha_im11)*PH/2
            ZT2S2 = (alpha_re22 + 1j*alpha_im22)*PH/2
            ZT1S2 = (alpha_re21 + 1j*alpha_im21)*PH/2 #note the apparent flip in subscript here; this is due to constraint subscript labeling going <S_i|P|T_j> and ZTS labeling going <T|ZTS|S>
            ZT2S1 = (alpha_re12 + 1j*alpha_im12)*PH/2
            ZTS[:num_basis,:num_basis] += ZT1S1
            ZTS[:num_basis,num_basis:] += ZT1S2
            ZTS[num_basis:,:num_basis] += ZT2S1
            ZTS[num_basis:,num_basis:] += ZT2S2
        
        #print('colormap of ZTS')
        #plt.figure()
        #plt.imshow(np.abs(ZTS))
        #plt.show()
        ZTS_S = ZTS @ Silist[mode]
        ZTS_S[:num_basis] += Gmat_od_list[mode].T.conj() @ (Salist[mode][:num_basis] - Silist[mode][:num_basis])
        ZTS_S[num_basis:] += Gmat_od_list[mode].T.conj() @ (Salist[mode][num_basis:] - Silist[mode][num_basis:])
        ZTS_Slist.append(ZTS_S)
    return ZTS_Slist


def get_2source_comm_normdiffsqr_gradZTS_Slist(Lags, Silist, Salist, Plistlist):
    gradZTS_Slist = []
    for mode in range(len(Silist)):
        Plist = Plistlist[mode]
        num_basis = Plist[0].shape[0]
        shape_ZTS = (2*num_basis,2*num_basis)
        gradZTS_S = []
        for i in range(len(Plist)):
            PH = Plist[i].T.conj()
            gradZTS_re11 = np.zeros(shape_ZTS, dtype=np.complex)
            gradZTS_re11[:num_basis,:num_basis] = PH / 2
            gradZTS_S_re11 = gradZTS_re11 @ Silist[mode]
            
            gradZTS_re22 = np.zeros(shape_ZTS, dtype=np.complex)
            gradZTS_re22[num_basis:,num_basis:] = PH / 2
            gradZTS_S_re22 = gradZTS_re22 @ Silist[mode]
            
            gradZTS_re12 = np.zeros(shape_ZTS, dtype=np.complex)
            gradZTS_re12[num_basis:,:num_basis] = PH / 2
            gradZTS_S_re12 = gradZTS_re12 @ Silist[mode]
            
            gradZTS_re21 = np.zeros(shape_ZTS, dtype=np.complex)
            gradZTS_re21[:num_basis,num_basis:] = PH / 2
            gradZTS_S_re21 = gradZTS_re21 @ Silist[mode]
            
            gradZTS_S.append(gradZTS_S_re11)
            gradZTS_S.append(gradZTS_S_re12)
            gradZTS_S.append(gradZTS_S_re21)
            gradZTS_S.append(gradZTS_S_re22)
            gradZTS_S.append(1j*gradZTS_S_re11)
            gradZTS_S.append(1j*gradZTS_S_re12)
            gradZTS_S.append(1j*gradZTS_S_re21)
            gradZTS_S.append(1j*gradZTS_S_re22)
        gradZTS_Slist.append(gradZTS_S)
    return gradZTS_Slist

def test_2source_comm_normdiffsqr_gradZTS_Slist(Lags, Silist, Salist, Plistlist, Gmat_od_list):
    ZTS_Slist_0 = get_2source_comm_normdiffsqr_ZTS_Slist(Lags, Silist, Salist, Plistlist, Gmat_od_list)
    delta = 1e-3*np.abs(Lags[0])
    Lags[0] += delta
    ZTS_Slist_1 = get_2source_comm_normdiffsqr_ZTS_Slist(Lags, Silist, Salist, Plistlist, Gmat_od_list)
    Lags[0] -= delta
    gradZTS_Slist = []
    for mode in range(len(Silist)):
        Plist = Plistlist[mode]
        num_basis = Plist[0].shape[0]
        shape_ZTS = (2*num_basis,2*num_basis)
            
        PH = Plist[0].T.conj()
        gradZTS_re11 = np.zeros(shape_ZTS, dtype=np.complex)
        gradZTS_re11[:num_basis,:num_basis] = PH / 2
        gradZTS_S_re11 = gradZTS_re11 @ Silist[mode]
        print('gradZTS_S_re11', gradZTS_S_re11)
        print('fdgradZTS_S_re11', (ZTS_Slist_1[mode]-ZTS_Slist_0[mode]) / delta)
            
    return gradZTS_Slist

#from dualbound.Lagrangian.spatialProjopt_2source_Zops_numpy import Z_TT, grad_Z_TT
#from dualbound.Lagrangian.spatialProjopt_2source_vecs_numpy import get_Tvec, get_Tvec_gradTvec
#from dualbound.Lagrangian.spatialProjopt_2source_dualgradHess_fullSvec_numpy import get_spatialProj_dualgrad_fullS
def get_sphere_planewave_xdipole_screening_numpy(k, dist, Ro, Rd_Plist, chi, xdfac,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 initLags=None, incPIm=False, incRegions=None, incCross=True,
                                                 mpdps=60, normtol=1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True,
                                                 opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    
    mp.dps=mpdps
    klim_update = get_sphere_planewave_xdipole_screening_datalists_numpy(k, dist, Ro, Rd_Plist, chi, xdfac,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
    
    modenum = len(Silist)
    ##############add in normsqr code!!!!!#####################
    normsqr_const1 = 0.0
    normsqr_Po_Si1 = 0.0
    normsqr_const2 = 0.0
    normsqr_Po_Si2 = 0.0
    for mode in range(modenum):
        Po = Po_list[mode]
        num_basis = Po.shape[0]
        Si1 = Silist[mode][:num_basis]
        Si2 = Silist[mode][num_basis:]
        Sa1 = Salist[mode][:num_basis]
        Sa2 = Salist[mode][num_basis:]
        normsqr_const1 += -np.real(np.vdot(Si1, Po @ Si1)) - np.real(np.vdot(Sa1, Po @ Sa1)) + 2*np.real(np.vdot(Sa1, Po @ Si1))
        normsqr_Po_Si1 += np.real(np.vdot(Po @ Si1, Po @ Si1))
        normsqr_const2 += -np.real(np.vdot(Si2, Po @ Si2)) - np.real(np.vdot(Sa2, Po @ Sa2)) + 2*np.real(np.vdot(Sa2, Po @ Si2))
        normsqr_Po_Si2 += np.real(np.vdot(Po @ Si2, Po @ Si2))
    
    Lagnum = 8 + 8*(len(Rd_Plist)-1)
    include = np.array([True]*Lagnum)
    if incRegions is None: #use default region inclusion: every region except the innermost
        include[8:16] = False
    else:
        for i in range(len(incRegions)):
            include[8+8*i:16+8*i] = incRegions[i]
    
    if not incPIm:
        for i in range(1,len(Rd_Plist)-1+1): #i starts from 1 because we always want to keep the full domain Im constraints
            include[8*i+4:8*i+8] = False
    
    if not incCross: #set all the cross constraints to 0
        for i in range(Lagnum):
            imod8 = i % 8
            if imod8==1 or imod8==2 or imod8==5 or imod8==6:
                include[i] = False
    
    """
    for i in range(Lagnum): #try with one of the cross constraints set to 0
        imod8 = i % 8
        if imod8==2 or imod8==6:
        #if imod8==1 or imod8==5:
            include[i] = False
    """
    
    if initLags is None:
        Lags = spatialProjopt_2source_find_feasiblept(Lagnum, include, Olist, UPlistlist)
    else:
        Lags = initLags.copy()
        Lags[4] = np.abs(Lags[4])+0.01
        Lags[7] = np.abs(Lags[7])+0.01
    
    validLagsfunc = lambda L: get_ZTT_mineig(L, Olist, UPlistlist, eigvals_only=True)[1] #[1] because we only need the minimum eigenvalue part of the returned values
    while validLagsfunc(Lags)<0:
        print('zeta1', Lags[4], 'zeta2', Lags[7])
        Lags[4] *= 1.5
        Lags[7] *= 1.5#enlarge zeta1 and zeta2 until we enter domain of validity
    print('initial Lags', Lags)
    
    ZTS_Slistfunc = lambda L, Slist: get_2source_comm_normdiffsqr_ZTS_Slist(L, Slist, Salist, Plistlist, Gmat_od_list)
    gradZTS_Slistfunc = lambda L, Slist: get_2source_comm_normdiffsqr_gradZTS_Slist(L, Slist, Salist, Plistlist)
    
    dgHfunc = lambda dof, dofgrad, dofHess, Slist, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fullS(dof, dofgrad, dofHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc,dualconst=normsqr_const1+normsqr_const2, get_grad=get_grad, get_Hess=get_Hess)

    #check_dgHfunc(Lags[include], Silist, dgHfunc)
    mineigfunc = lambda dof, eigvals_only=False: get_inc_ZTT_mineig(dof, include, Olist, UPlistlist, eigvals_only=eigvals_only)
    
    opt_incLags, opt_incgrad, opt_dual, opt_obj = modS_opt(Lags[include], Silist, dgHfunc, mineigfunc, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
    optLags = np.zeros(Lagnum)
    optgrad = np.zeros(Lagnum)
    optLags[include] = opt_incLags
    optgrad[include] = opt_incgrad
    print('the remaining constraint violations')
    print(optgrad)
    
    optdual1, optdual2 = get_2source_separate_duals(optLags, Olist, Plistlist, UPlistlist, Silist, ZTS_Slistfunc, dualconst1=normsqr_const1, dualconst2 = normsqr_const2, include=include)
    optobj1, optobj2 = get_2source_separate_obj(optLags, Olist, Plistlist, UPlistlist, Silist, ZTS_Slistfunc, dualconst1=normsqr_const1, dualconst2 = normsqr_const2, include=include)
    print('the separate duals:', optdual1, optdual2)
    print('the separate obj:', optobj1, optobj2)
    print('the separate observation Si norms:', normsqr_Po_Si1, normsqr_Po_Si2)
    print('the separate normsqrratios:', -optdual1/normsqr_Po_Si1, -optdual2/normsqr_Po_Si2)



def get_sphere_planewave_xdipole_equalnorm_screening_numpy(k, dist, Ro, Rd_Plist, chi, 
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 initLags=None, incPIm=False, incRegions=None, incCross=True,
                                                 mpdps=60, normtol=1e-4, gridpts=1000, maxveclim=40, Minitklim=22, Ninitklim=22, Taylor_tol=1e-12, Unormtol=1e-6, includeI=True,
                                                 opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    
    #np.seterr(under='raise', invalid='raise')
    #np.seterr(all='raise')
    mp.dps=mpdps
    klim_update = get_sphere_planewave_xdipole_screening_datalists_numpy(k, dist, Ro, Rd_Plist, chi, 1.0,
                                                 Silist, Salist, Olist, Gmat_od_list, Po_list,
                                                 Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                 wig_1llp1_000,wig_1llm1_000, wig_1llp1_1m10,wig_1llm1_1m10, 
                                                 mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol)
    
    modenum = len(Silist)
    ##############add in normsqr code!!!!!#####################
    normsqr_const1 = 0.0
    normsqr_Po_Si1 = 0.0
    normsqr_const2 = 0.0
    normsqr_Po_Si2 = 0.0
    for mode in range(modenum):
        Po = Po_list[mode]
        num_basis = Po.shape[0]
        Si1 = Silist[mode][:num_basis]
        Si2 = Silist[mode][num_basis:]
        Sa1 = Salist[mode][:num_basis]
        Sa2 = Salist[mode][num_basis:]
        normsqr_const1 += -np.real(np.vdot(Si1, Po @ Si1)) - np.real(np.vdot(Sa1, Po @ Sa1)) + 2*np.real(np.vdot(Sa1, Po @ Si1))
        normsqr_Po_Si1 += np.real(np.vdot(Po @ Si1, Po @ Si1))
        normsqr_Po_Si2 += np.real(np.vdot(Po @ Si2, Po @ Si2))
    
    #scale Si2list so that the norms of the two incident fields over the observation region are the same
    xdfac = np.sqrt(normsqr_Po_Si1/normsqr_Po_Si2)
    normsqr_const2 = 0.0
    normsqr_Po_Si2 = 0.0 #after rescaling, reevaluate normsqr_const2 and normsqr_Po_Si2
    for mode in range(modenum):
        Po = Po_list[mode]
        num_basis = Po.shape[0]
        Silist[mode][num_basis:] *= xdfac
        Si2 = Silist[mode][num_basis:]
        Sa2 = Salist[mode][num_basis:]
        normsqr_const2 += -np.real(np.vdot(Si2, Po @ Si2)) - np.real(np.vdot(Sa2, Po @ Sa2)) + 2*np.real(np.vdot(Sa2, Po @ Si2))
        normsqr_Po_Si2 += np.real(np.vdot(Po @ Si2, Po @ Si2))
    
    Lagnum = 8 + 8*(len(Rd_Plist)-1)
    include = np.array([True]*Lagnum)
    if incRegions is None: #use default region inclusion: every region except the innermost
        include[8:16] = False
    else:
        for i in range(len(incRegions)):
            include[8+8*i:16+8*i] = incRegions[i]
    
    if not incPIm:
        for i in range(1,len(Rd_Plist)-1+1): #i starts from 1 because we always want to keep the full domain Im constraints
            include[8*i+4:8*i+8] = False
    
    if not incCross: #set all the cross constraints to 0
        for i in range(Lagnum):
            imod8 = i % 8
            if imod8==1 or imod8==2 or imod8==5 or imod8==6:
                include[i] = False
    
    """
    for i in range(Lagnum): #try with one of the cross constraints set to 0
        imod8 = i % 8
        if imod8==2 or imod8==6:
        #if imod8==1 or imod8==5:
            include[i] = False
    """
    
    if initLags is None:
        Lags = spatialProjopt_2source_find_feasiblept(Lagnum, include, Olist, UPlistlist)
    else:
        Lags = initLags.copy()
        Lags[4] = np.abs(Lags[4])+0.01
        Lags[7] = np.abs(Lags[7])+0.01
    
    validLagsfunc = lambda L: get_ZTT_mineig(L, Olist, UPlistlist, eigvals_only=True)[1] #[1] because we only need the minimum eigenvalue part of the returned values
    while validLagsfunc(Lags)<0:
        print('zeta1', Lags[4], 'zeta2', Lags[7])
        Lags[4] *= 1.5
        Lags[7] *= 1.5 #enlarge zeta1 and zeta2 until we enter domain of validity
    print('initial Lags', Lags)
    
    ZTS_Slistfunc = lambda L, Slist: get_2source_comm_normdiffsqr_ZTS_Slist(L, Slist, Salist, Plistlist, Gmat_od_list)
    gradZTS_Slistfunc = lambda L, Slist: get_2source_comm_normdiffsqr_gradZTS_Slist(L, Slist, Salist, Plistlist)
    
    dgHfunc = lambda dof, dofgrad, dofHess, Slist, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fullS(dof, dofgrad, dofHess, include, Olist, Plistlist, UPlistlist, Slist, ZTS_Slistfunc, gradZTS_Slistfunc,dualconst=normsqr_const1+normsqr_const2, get_grad=get_grad, get_Hess=get_Hess)

    #check_dgHfunc(Lags[include], Silist, dgHfunc)
    mineigfunc = lambda dof, eigvals_only=False: get_inc_ZTT_mineig(dof, include, Olist, UPlistlist, eigvals_only=eigvals_only)
    
    opt_incLags, opt_incgrad, opt_dual, opt_obj = modS_opt(Lags[include], Silist, dgHfunc, mineigfunc, opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
    optLags = np.zeros(Lagnum)
    optgrad = np.zeros(Lagnum)
    optLags[include] = opt_incLags
    optgrad[include] = opt_incgrad
    print('the remaining constraint violations')
    print(optgrad)
    
    optdual1, optdual2 = get_2source_separate_duals(optLags, Olist, Plistlist, UPlistlist, Silist, ZTS_Slistfunc, dualconst1=normsqr_const1, dualconst2 = normsqr_const2, include=include)
    optobj1, optobj2 = get_2source_separate_obj(optLags, Olist, Plistlist, UPlistlist, Silist, ZTS_Slistfunc, dualconst1=normsqr_const1, dualconst2 = normsqr_const2, include=include)
    print('the separate duals:', optdual1, optdual2)
    print('the separate obj:', optobj1, optobj2)
    print('the separate observation Si norms:', normsqr_Po_Si1, normsqr_Po_Si2)
    print('xdfac is', xdfac)
    totnormsqrratio = -opt_dual / (normsqr_Po_Si1 + normsqr_Po_Si2)
    normsqrratio1 = -optdual1/normsqr_Po_Si1
    normsqrratio2 = -optdual2/normsqr_Po_Si2
    print('the separate normsqrratios:', normsqrratio1, normsqrratio2)
    return optLags, -opt_dual, -opt_obj, totnormsqrratio, -optdual1, -optdual2, normsqr_Po_Si1, normsqr_Po_Si2, normsqrratio1, normsqrratio2, klim_update


def sweep_norm_remainder_spherical_planewave_xdipole_equalnorm_screening_varydist(k, distlist, Ro, Rd_Plist, chi, 
                                                              incPIm=False, incRegions=None, incCross=True, 
                                                              normtol = 1e-4, mpdps=60, maxveclim=40, gridpts=1000, Minitklim=25, Ninitklim=25, 
                                                              Taylor_tol=1e-12, Unormtol=1e-6, opttol=1e-2, modSratio=1e-2, check_iter_period=20,
                                                              filename=None, feedforward=True):
    
    #for now let the external calling script set the domains
    flag = False
    if not (filename is None):
        flag = True
        outdist = open(filename+'_distlist.txt','w')
        outtotdual = open(filename+'_totduallist.txt','w')
        outtotobj = open(filename+'_totobjlist.txt','w')
        outtotratio = open(filename+'_totnormsqrratiolist.txt', 'w')
        outdual1 = open(filename+'_dual1list.txt','w')
        outdual2 = open(filename+'_dual2list.txt','w')
        outSi1normsqr = open(filename+'_Si1normsqrlist.txt','w')
        outSi2normsqr = open(filename+'_Si2normsqrlist.txt','w')
        outratio1 = open(filename+'_normsqrratio1list.txt', 'w')
        outratio2 = open(filename+'_normsqrratio2list.txt', 'w')
        
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
    
    totduallist = []
    totobjlist = []
    dual1list = []
    dual2list = []
    currentdistlist = []
    Si1normsqrlist = []
    Si2normsqrlist = []
    totnormsqrratiolist = []
    normsqrratio1list = []
    normsqrratio2list = []
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
                Lags, totdual, totobj, totnormsqrratio, dual1, dual2, Si1normsqr, Si2normsqr, normsqrratio1, normsqrratio2, klim = get_sphere_planewave_xdipole_equalnorm_screening_numpy(k, dist, Ro, Rd_Plist, chi, 
                                                                                                                   [], [], Olist, Gmat_od_list, Po_list,
                                                                                                                   Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                                                                                   wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10,                                                                
                                                                                                                   initLags=None, incPIm=incPIm, incRegions=incRegions, incCross=incCross,
                                                                                                                   mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol,
                                                                                                                   opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
                        
            else: #feed previous optimal parameters as starting point for next optimization
                Minitklim = max(Minitklim, klim+3); Ninitklim = max(Ninitklim,klim+3) #change Minitklim based on required size from previous iteration
                print('new Minitklim is', Minitklim, 'new Ninitklim is', Ninitklim)
                if Minitklim > mp.dps:
                    mp.dps = Minitklim
                Lags, totdual, totobj, totnormsqrratio, dual1, dual2, Si1normsqr, Si2normsqr, normsqrratio1, normsqrratio2, klim = get_sphere_planewave_xdipole_equalnorm_screening_numpy(k, dist, Ro, Rd_Plist, chi, 
                                                                                                                   [], [], Olist, Gmat_od_list, Po_list,
                                                                                                                   Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                                                                                   wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10, 
                                                                                                                   initLags=Lags, incPIm=incPIm, incRegions=incRegions, incCross=incCross,
                                                                                                                   mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol,
                                                                                                                   opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
        except FloatingPointError:
            print('Encounter with ill-conditioned ZTT / bounds approaching 0, will now try no feedforward default init', flush=True)
            try: #try again with default initialization
                Lags, totdual, totobj, totnormsqrratio, dual1, dual2, Si1normsqr, Si2normsqr, normsqrratio1, normsqrratio2, klim = get_sphere_planewave_xdipole_equalnorm_screening_numpy(k, dist, Ro, Rd_Plist, chi, 
                                                                                                                   [], [], Olist, Gmat_od_list, Po_list,
                                                                                                                   Plistlist, UPlistlist, Rgnormlistlist, subbasis_indlistlist, 
                                                                                                                   wig_1llp1_000, wig_1llm1_000, wig_1llp1_1m10, wig_1llm1_1m10,                                                                
                                                                                                                   initLags=None, incPIm=incPIm, incRegions=incRegions, incCross=incCross,
                                                                                                                   mpdps=mpdps, normtol=normtol, gridpts=gridpts, maxveclim=maxveclim, Minitklim=Minitklim, Ninitklim=Ninitklim, Taylor_tol=Taylor_tol, Unormtol=Unormtol,
                                                                                                                   opttol=opttol, modSratio=modSratio, check_iter_period=check_iter_period)
            except: #if all else fails, skip to next dist point
                print('all attempts at finding optimum at dist=', dist, 'have failed')
                print('moving on to next point', flush=True)
                continue
        except (KeyboardInterrupt, SystemExit): #in case I am interrupting by hand
            raise
            
        totduallist.append(totdual)
        totobjlist.append(totobj)
        totnormsqrratiolist.append(totnormsqrratio)
        currentdistlist.append(dist)
        dual1list.append(dual1)
        dual2list.append(dual2)
        Si1normsqrlist.append(Si1normsqr)
        Si2normsqrlist.append(Si2normsqr)
        normsqrratio1list.append(normsqrratio1)
        normsqrratio2list.append(normsqrratio2)
        
        if flag:
            outtotdual.write(str(totdual)+'\n'); outtotdual.flush()
            outtotobj.write(str(totobj)+'\n'); outtotobj.flush()
            outtotratio.write(str(totnormsqrratio)+'\n'); outtotratio.flush()
            outdual1.write(str(dual1)+'\n'); outdual1.flush()
            outdual2.write(str(dual2)+'\n'); outdual2.flush()
            outSi1normsqr.write(str(Si1normsqr)+'\n'); outSi1normsqr.flush()
            outSi2normsqr.write(str(Si2normsqr)+'\n'); outSi2normsqr.flush()
            outratio1.write(str(normsqrratio1)+'\n'); outratio1.flush()
            outratio2.write(str(normsqrratio2)+'\n'); outratio2.flush()
            
            np.save(filename+'_distlist.npy', np.array(currentdistlist))
            np.save(filename+'_totduallist.npy', np.array(totduallist))
            np.save(filename+'_totobjlist.npy', np.array(totobjlist))
            np.save(filename+'_dual1list.npy', np.array(dual1list))
            np.save(filename+'_dual2list.npy', np.array(dual2list))
            np.save(filename+'_Si1normsqrlist.npy', np.array(Si1normsqrlist))
            np.save(filename+'_Si2normsqrlist.npy', np.array(Si2normsqrlist))
            np.save(filename+'_normsqrratio1list.npy', np.array(normsqrratio1list))
            np.save(filename+'_normsqrratio2list.npy', np.array(normsqrratio2list))
        
    if flag:
        outdist.close()
        outtotdual.close()
        outtotobj.close()
        outtotratio.close()
        outdual1.close()
        outdual2.close()
        outSi1normsqr.close()
        outSi2normsqr.close()
        outratio1.close()
        outratio2.close()
    
    return currentdistlist, totduallist, totobjlist, totnormsqrratiolist, dual1list, dual2list, Si1normsqrlist, Si2normsqrlist, normsqrratio1list, normsqrratio2list


def check_dgHfunc(dof, Slist, dgHfunc):
    dofnum = len(dof)
    dualfunc = lambda L: dgHfunc(L, [],[], Slist, get_grad=False, get_Hess=False)
    
    grad = np.zeros(dofnum)
    Hess = np.zeros((dofnum, dofnum))
    P0 = dgHfunc(dof,grad,Hess,Slist)
    print('P0 is', P0)

    step = 1e-3
    
    for i in range(dofnum):
        delta = step*np.abs(dof[i])
        if delta<1e-10:
            continue
        dof[i] += delta
        P1 = dualfunc(dof)
        #P1 = dgHfunc(dof, delgrad, delHess, Slist)
        dof[i] -= delta
        print('P1',P1,'P0',P0,'delta',delta)
        fdgrad = (P1-P0)/delta
        print('for alpha'+str(i)+' the calculated gradient and fd estimate are', grad[i], fdgrad)
    
    
    for i in range(dofnum):
        delta_i = step*np.abs(dof[i])
        if delta_i<1e-10:
            continue
        pidof = dof.copy(); pidof[i] += delta_i
        midof = dof.copy(); midof[i] -= delta_i
        
        fdsqr_i = (dualfunc(pidof)-2*P0+dualfunc(midof))/delta_i**2
        print('for alpha'+str(i),'the calculated 2nd derivative and fd estimate are',Hess[i,i],fdsqr_i)
        
        for j in range(i,dofnum):
            delta_j = step*np.abs(dof[j])
            if delta_j<1e-10:
                continue
            pipjdof = dof.copy(); pipjdof[i] += delta_i; pipjdof[j] += delta_j
            pimjdof = dof.copy(); pimjdof[i] += delta_i; pimjdof[j] -= delta_j
            mipjdof = dof.copy(); mipjdof[i] -= delta_i; mipjdof[j] += delta_j
            mimjdof = dof.copy(); mimjdof[i] -= delta_i; mimjdof[j] -= delta_j
            
            fd2ij = (dualfunc(pipjdof)-dualfunc(pimjdof)-dualfunc(mipjdof)+dualfunc(mimjdof))/(4*delta_i*delta_j)
            print('for alpha'+str(i),'and', 'for alpha'+str(j), 'the calculated cross derivative and fdestimate are', Hess[i,j], fd2ij)
        
    testSym = Hess - Hess.T
    print('test Hessian symmetry', np.linalg.norm(testSym))
    
    eigw, eigH = np.linalg.eigh(Hess)
    print('test Hess PD', eigw)
    print(eigH)
    
