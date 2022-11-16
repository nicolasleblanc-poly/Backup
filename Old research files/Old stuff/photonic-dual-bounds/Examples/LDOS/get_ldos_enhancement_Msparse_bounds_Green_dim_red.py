import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys, time

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_multiSource_Msparse_gradZTS_S, get_Msparse_gradZTS_S, get_multiSource_Msparse_gradZTT, get_Msparse_gradZTT, get_inc_ZTT_mineig, get_inc_PD_ZTT_mineig, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity

from dualbound.Lagrangian.spatialProjopt_vecs_Msparse import get_ZTTcho_Tvec

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept

from dualbound.Optimization.BFGS_fakeSource import BFGS_fakeS
from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart


from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_Gddinv
from dualbound.Rect.rect_domains import get_rect_proj_in_rect_region, divide_rect_region
from dualbound.Rect.rect_iterative_splitting import dualopt_Msparse_iterative_splitting
from dualbound.Rect.mask_iterative_splitting import split_sparseP_by_violation, plot_sparseP_distribution, get_max_violation_sparseP, get_max_violation_sparseP_2, normalize_P,reduce_P


def get_Msparse_bound(chi, Si, O_lin, O_quad, Ginv, Plist, include, dualconst=0.0, initLags=None, getViolation=False, opttol=1e-2, fakeSratio=1e-2, iter_period=1000):
    
    ######generate GinvconjPlist and UPlist
    GinvconjPlist = []
    UPlist = []
    for i in range(len(Plist)):
        GinvconjP = Ginv.conj().T @ Plist[i].conj().T
        UMP = (Ginv.conj().T @ GinvconjP.conj().T)/np.conj(chi) - GinvconjP.conj().T
        GinvconjPlist.append(GinvconjP.tocsc())
        UPlist.append(UMP.tocsc())

    gradZTT = get_multiSource_Msparse_gradZTT(1, UPlist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
    gradZTS_S = get_multiSource_Msparse_gradZTS_S(1, Si, GinvconjPlist)

    dgfunc = lambda dof, dofgrad, fSl, get_grad=True: get_inc_spatialProj_dualgrad_fakeS_Msparse(dof, dofgrad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, chofac=ZTTchofac)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)

    mineigfunc = lambda dof: get_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    if initLags is None:
        Lags = spatialProjopt_find_feasiblept(len(include), include, O_quad, gradZTT)
    else:
        Lags = initLags.copy()

    print('Lags', Lags)

    while True:
        tmp = check_spatialProj_Lags_validity(Lags, O_quad, gradZTT)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[1] *= 1.5

    optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    
    
    optLags = np.zeros(len(include), dtype=np.double)
    optLags[include] = optincLags[:]
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]

    print('the remaining constraint violations')
    print(optgrad)

    if getViolation:
        ZTS_S = O_lin.copy()
        ZTT = O_quad.copy()
        for i in range(len(optLags)):
            ZTS_S += optLags[i] * gradZTS_S[i]
            ZTT += optLags[i] * gradZTT[i]

        _, optGT = get_ZTTcho_Tvec(ZTT, ZTS_S, ZTTchofac)

        optT = Ginv @ optGT
        UdagT = optT/chi - optGT
        violation = np.conj(Si)*optT - np.conj(UdagT)*optT
        print('shape of violation', violation.shape)
        return optLags, optgrad, dualval, objval, violation
    
    else:
        return optLags, optgrad, dualval, objval


def get_Msparse_1designrect_normal_absorption(chi, wvlgth, design_x, design_y, pml_sep, pml_thick, gpr, NProjx, NProjy, initLags=None, fakeSratio=1e-2, iter_period=1000):

    zinv = np.imag(chi) / np.real(chi*np.conj(chi))
    
    domain_x = 2*pml_sep + design_x
    domain_y = 2*pml_sep + design_y

    dl = 1.0 / gpr
    nonpmlNx = int(np.round(domain_x / dl))
    nonpmlNy = int(np.round(domain_y / dl))
    Mx = int(np.round(design_x / dl))
    My = int(np.round(design_y / dl))
    Npml = int(np.round(pml_thick / dl))
    Npml_sep = int(np.round(pml_sep / dl))
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    design_ulx = (Nx-Mx)//2
    design_uly = (Ny-My)//2
    design_ulco = (design_ulx,design_uly)
    design_brco = (design_ulx+Mx,design_uly+My)

    designMask = np.zeros((Nx,Ny), dtype=np.bool)
    designMask[design_ulx:design_ulx+Mx,design_uly:design_uly+My] = True

    proj_ulco_list = [design_ulco]
    proj_brco_list = [design_brco]

    subproj_ulco_list, subproj_brco_list = divide_rect_region(design_ulco, design_brco, NProjx, NProjy)
    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:])
    num_proj = len(proj_ulco_list)
    

    print('Nx', Nx, 'Ny', Ny, 'Npml', Npml)
    Gddinv, _ = get_Gddinv(wvlgth, dl, Nx, Ny, Npml, designMask)

    print('Gddinv format', Gddinv.format, 'Gddinv shape', Gddinv.shape, 'Gddinv elements', Gddinv.count_nonzero(), flush=True)
    
    #first test with global constraints only
    UM = (Gddinv.conj().T @ Gddinv)/np.conj(chi) - Gddinv

    AsymUM = (UM - UM.conj().T) / 2j
    print('UM format', UM.format, 'AsymUM format', AsymUM.format, flush=True)

    Plist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region(design_ulco, design_brco, proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        Plist.append(Proj_mat)

    include = [True,True] * len(proj_brco_list)

    
    ###setting up normal incident planewave source###
    xgrid = np.linspace(-domain_x/2.0, domain_x/2.0, nonpmlNx)
    ygrid = np.linspace(-domain_y/2.0, domain_y/2.0, nonpmlNy)
    xgrid,ygrid = np.meshgrid(xgrid,ygrid, indexing='ij')
    k0 = 2*np.pi/wvlgth
    phasegrid = k0*xgrid #normal incidence
    Si_grid = np.exp(1j*phasegrid)
    Si_desvec = Si_grid[(nonpmlNx-Mx)//2:(nonpmlNx-Mx)//2+Mx,(nonpmlNy-My)//2:(nonpmlNy-My)//2+My].flatten()
        
    O_lin = np.zeros_like(Si_desvec, dtype=np.complex)

    O_quad = (-zinv * dl**2 * (Gddinv.conj().T @ Gddinv)).tocsc()


    optLags, optgrad, dualval, objval, violation = get_Msparse_bound(chi, Si_desvec, O_lin, O_quad, Gddinv, Plist, include, getViolation=True, fakeSratio=fakeSratio, iter_period=iter_period)

    print('check violation correctness')
    for i in range(len(Plist)):
        Recstrt = np.sum(Plist[i] @ violation)
        print('optgrad Recstrt', optgrad[2*i])
        print('violation Recstrt', Recstrt)
    
    sigma_abs = k0 * dualval
    sigma_enh = sigma_abs / design_y
    return optLags, optgrad, dualval, objval, sigma_abs, sigma_enh


def get_Msparse_1designrect_normal_absorption_violation_splitting(chi, wvlgth, design_x, design_y, vacuum_x, vacuum_y, emitter_x, emitter_y, pml_sep, pml_thick, gpr, initLags=None, fakeSratio=1e-2, tol_lindep=1e-6, tol_dualchange=1e-3, iter_period=1000, name='test',len_sp_name='lsp.txt',sp_root_name='sparse_p',lags_name='Lags.txt'):

    zinv = np.imag(chi) / np.real(chi*np.conj(chi))
    
    domain_x = 2*pml_sep + design_x
    domain_y = 2*pml_sep + design_y

    dl = 1.0 / gpr
    nonpmlNx = int(np.round(domain_x / dl))
    nonpmlNy = int(np.round(domain_y / dl))
    Mx = int(np.round(design_x / dl))
    My = int(np.round(design_y / dl))
    vacuumx = int(np.round(vacuum_x / dl))
    vacuumy = int(np.round(vacuum_y / dl))
    emitterx = int(np.round(emitter_x / dl))
    emittery = int(np.round(emitter_y / dl))
    Npml = int(np.round(pml_thick / dl))
    Npml_sep = int(np.round(pml_sep / dl))
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    design_ulx = (Nx-Mx)//2
    design_uly = (Ny-My)//2
    design_ulco = (design_ulx,design_uly)
    design_brco = (design_ulx+Mx,design_uly+My)

    designMask = np.zeros((Nx,Ny), dtype=np.bool)
    designMask[design_ulx:design_ulx+Mx,design_uly:design_uly+My] = True
    designMask[design_ulx+(Mx-vacuumx)//2:design_ulx+(Mx-vacuumx)//2+vacuumx,design_uly+(My-vacuumy)//2:design_uly+(My-vacuumy)//2+vacuumy] = False
    

    print('Nx', Nx, 'Ny', Ny, 'Npml', Npml)
    Gddinv, _ = get_Gddinv(wvlgth, dl, Nx, Ny, Npml, designMask)
    
    print('Gddinv format', Gddinv.format, 'Gddinv shape', Gddinv.shape, 'Gddinv elements', Gddinv.count_nonzero(), flush=True)
    
    #first test with global constraints only
    UM = (Gddinv.conj().T @ Gddinv)/np.conj(chi) - Gddinv

    AsymUM = (UM - UM.conj().T) / 2j
    print('UM format', UM.format, 'AsymUM format', AsymUM.format, flush=True)

    Id = sp.eye(Gddinv.shape[0], format="csc")
    Plist = [Id.copy()]
    Lags = None
    ###setting up normal incident planewave source###
    xgrid = np.linspace(-domain_x/2.0, domain_x/2.0, nonpmlNx)
    ygrid = np.linspace(-domain_y/2.0, domain_y/2.0, nonpmlNy)
    xgrid,ygrid = np.meshgrid(xgrid,ygrid, indexing='ij')
    k0 = 2*np.pi/wvlgth
    phasegrid = k0*xgrid #normal incidence
    
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    Ezfield = np.zeros((Nx,Ny),dtype=np.complex)
    vacPrad = 0
    for k in range(Ny//2+int(max([emittery//2,np.ceil(emittery/2)])),Ny//2-emittery//2,-1):
        for ii in range(Nx//2+int(max([emitterx//2,np.ceil(emitterx/2)])),Nx//2-emitterx//2,-1):
            cx = ii
            cy = k #position of dipole in entire computational domain
            Ezfield_temp = get_TM_dipole_field(wvlgth, dl, Nx, Ny, cx, cy, Npml) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
            Ezfield += Ezfield_temp
    for k in range(Ny//2+int(max([emittery//2,np.ceil(emittery/2)])),Ny//2-emittery//2,-1):
        for ii in range(Nx//2+int(max([emitterx//2,np.ceil(emitterx/2)])),Nx//2-emitterx//2,-1):
            cx = ii
            cy = k
            vacPrad_temp = -0.5 * np.real(Ezfield[cx,cy]) #default unit amplitude dipole
            vacPrad += vacPrad_temp
    S1 = (k0/1j/Z) * Ezfield[designMask] #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dl**2 #same dl**2 factor as above

    Si_grid = S1
    Si_desvec = Si_grid.flatten() 

    O_lin = ((Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dl**2).flatten() 

    dualval0 = -10
    tt = 0
    try:
        len_sp = np.loadtxt(len_sp_name+str(np.imag(chi)))
        len_sp = int(len_sp)
        subPlist = []
        for ll in range(len_sp):
            spl = sp.load_npz(sp_root_name+str(ll)+str(np.imag(chi))+'.npz')
            subPlist += [spl]
        Lags = np.loadtxt(lags_name+str(np.imag(chi)))
        Lags = Lags[:2]
    except:
        pass
    print('Lags', Lags)
    include = [True,True] * len(Plist)
    optLags, optgrad, dualval, objval, violation = get_Msparse_bound(chi, Si_desvec, O_lin, O_quad, Gddinv, Plist, include, initLags=Lags, getViolation=True, fakeSratio=fakeSratio, iter_period=iter_period)
    Pdiags = np.atleast_2d(Id.diagonal())
    tt = 0
    while True:
        try:
            print('Lags', Lags)
            maxvP = get_max_violation_sparseP_2(violation)
            Plist.append(maxvP)
            print('\n', 'total number of projection constraints', len(Plist))
        
            Pdiags = np.concatenate((Pdiags, np.atleast_2d(maxvP.diagonal())))
            #find min svd of Pdiags to check linear dependence of all projection constraints
            _, svPdiags, _ = la.svd(Pdiags)
            print('minimum singular value for Pdiags', svPdiags[-1])
            include.extend([True,True]) #add in new max violation P
            initLags = np.zeros(len(include))
            initLags[:-2] = optLags[:] #start off with the new multipliers at exactly 0            include = [True,True] * len(Plist)
            optLags, optgrad, dualval, objval, violation = get_Msparse_bound(chi, Si_desvec, O_lin, O_quad, Gddinv, Plist, include, initLags=initLags, getViolation=True, fakeSratio=fakeSratio, iter_period=iter_period)
            
            print('check violation correctness',flush=True)
            for i in range(len(Plist)):
                Recstrt = np.sum(Plist[i] @ violation)
                print('optgrad Recstrt', optgrad[2*i])
                print('violation Recstrt', Recstrt)

            sigma_abs = k0 * dualval
            sigma_enh = sigma_abs / design_y
            print('sigma_abs', sigma_abs)
            print('sigma_enh', sigma_enh,flush=True)
            if tt%1 == 0:
                ind = 0
                include = include[:2*(ind+1)] #add in new max violation P
                Plist2,optLags2 = reduce_P(Plist,optLags,ind)
                Psym = (Plist2[ind]+Plist2[ind].conj().T)/2
                Pasym = (Plist2[ind]-Plist2[ind].conj().T)/(2j)
                P1sym = (Plist[ind]+Plist[ind].conj().T)/2
                P1asym = (Plist[ind]-Plist[ind].conj().T)/(2j)
                P2sym = (Plist[ind+1]+Plist[ind+1].conj().T)/2
                P2asym = (Plist[ind+1]-Plist[ind+1].conj().T)/(2j)
                LHS1 = optLags2[2*ind]*Psym+optLags2[2*ind+1]*Pasym
                RHS1 = optLags[2*ind]*P1sym+optLags[2*ind+1]*P1asym + optLags[2*(ind+1)]*P2sym + optLags[2*(ind+1)+1]*P2asym
                LHS2 = optLags2[2*ind+1]*Psym-optLags2[2*ind]*Pasym
                RHS2 = optLags[2*ind+1]*P1sym-optLags[2*ind]*P1asym + optLags[2*(ind+1)+1]*P2sym - optLags[2*(ind+1)]*P2asym
                print("Max real operator equality difference is",np.max(np.absolute(LHS1-RHS1)))
                print("Max imaginary operator equality difference is",np.max(np.absolute(LHS2-RHS2)),flush=True)
                GinvconjPlist = []
                UPlist = []
                GinvconjPlist2 = []
                UPlist2 = []
                for i in range(len(Plist)):
                    GinvconjP = Gddinv.conj().T @ Plist[i].conj().T
                    UMP = (Gddinv.conj().T @ GinvconjP.conj().T)/np.conj(chi) - GinvconjP.conj().T
                    GinvconjPlist.append(GinvconjP.tocsc())
                    UPlist.append(UMP.tocsc())
                for i in range(len(Plist2)):
                    GinvconjP2 = Gddinv.conj().T @ Plist2[i].conj().T
                    UMP2 = (Gddinv.conj().T @ GinvconjP2.conj().T)/np.conj(chi) - GinvconjP2.conj().T
                    GinvconjPlist2.append(GinvconjP2.tocsc())
                    UPlist2.append(UMP2.tocsc())

                gradZTT = get_multiSource_Msparse_gradZTT(1, UPlist)
                gradZTS_S = get_multiSource_Msparse_gradZTS_S(1, Si_desvec, GinvconjPlist)
                gradZTT2 = get_multiSource_Msparse_gradZTT(1, UPlist2)
                gradZTS_S2 = get_multiSource_Msparse_gradZTS_S(1, Si_desvec, GinvconjPlist2)

                ZTS_S = O_lin.copy()
                ZTT = O_quad.copy()
                ZTS_S2 = O_lin.copy()
                ZTT2 = O_quad.copy()
                for i in range(len(optLags)):
                    ZTS_S += optLags[i] * gradZTS_S[i]
                    ZTT += optLags[i] * gradZTT[i]
                for i in range(len(optLags2)):
                    ZTS_S2 += optLags2[i] * gradZTS_S2[i]
                    ZTT2 += optLags2[i] * gradZTT2[i]
                ZTS_tot = ZTS_S+ZTS_S.conj().T
                ZTS_tot2 = ZTS_S2+ZTS_S2.conj().T
                print("Max ZTS difference is",np.max(np.absolute(ZTS_tot-(ZTS_tot2))))
                print("Max ZTT difference is",np.max(np.absolute(ZTT-ZTT2)),flush=True)
                Plist = Plist2.copy()
                optLags = optLags2.copy()
                if np.absolute((dualval-dualval0)/dualval) < tol_dualchange:
                    break
                else:
                    dualval0 = dualval
                if svPdiags[-1] < tol_lindep:
                    break
            tt += 1
        except ValueError:
            break
    Prad_enh = dualval / vacPrad
    return Prad_enh


