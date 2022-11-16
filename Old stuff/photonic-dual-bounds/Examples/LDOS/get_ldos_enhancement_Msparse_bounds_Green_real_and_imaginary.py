import numpy as np
#import matplotlib.pyplot as plt
#from matplotlib.colors import DivergingNorm
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
from dualbound.Rect.mask_iterative_splitting import split_sparseP_by_violation, plot_sparseP_distribution


def get_Msparse_bound(chi, Si, O_lin, O_quad, Ginv, Plist, include, dualconst=0.0, initLags=None, getViolation=False, opttol=1e-2, fakeSratio=1e-2, iter_period=1000):

    ######generate GinvconjPlist and UPlist
    GinvconjPlist = []
    UPlist = []
    for i in range(len(Plist)):
        GinvconjP = Ginv.conj().T @ Plist[i]
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

    #optincLags, optincgrad, dualval, objval = BFGS_fakeS(Lags[include], Si, dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
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
        return optLags, optgrad, dualval, objval, np.real(violation), np.imag(violation)
    
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


    optLags, optgrad, dualval, objval, violation_r, violation_im = get_Msparse_bound(chi, Si_desvec, O_lin, O_quad, Gddinv, Plist, include, getViolation=True, fakeSratio=fakeSratio, iter_period=iter_period)

    print('check violation correctness')
    for i in range(len(Plist)):
        Recstrt = np.sum(Plist[i] @ violation)
        print('optgrad Recstrt', optgrad[2*i])
        print('violation Recstrt', Recstrt)
    
    sigma_abs = k0 * dualval
    sigma_enh = sigma_abs / design_y
    return optLags, optgrad, dualval, objval, sigma_abs, sigma_enh


def get_Msparse_1designrect_normal_absorption_violation_splitting(chi, wvlgth, design_x, design_y, vacuum_x, vacuum_y, emitter_x, emitter_y, pml_sep, pml_thick, gpr, initLags=None, fakeSratio=1e-2, iter_period=1000, name='test',len_sp_name='lsp.txt',sp_root_name='sparse_p',lags_name='Lags.txt'):

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
    subPlist = [Id.copy()]
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
            cy = k
            #cx = Npml + Npmlsep
            #cy = Ny//2 #position of dipole in entire computational domain
            Ezfield_temp = get_TM_dipole_field(wvlgth, dl, Nx, Ny, cx, cy, Npml) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
            Ezfield += Ezfield_temp
            vacPrad_temp = -0.5 * np.real(Ezfield_temp[cx,cy]) #default unit amplitude dipole
            vacPrad += vacPrad_temp
    S1 = (k0/1j/Z) * Ezfield[designMask] #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    #O_lin = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dl**2
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dl**2 #same dl**2 factor as above

    #Si_grid = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dl**2 #*np.ones_like(phasegrid)# exp(1j*phasegrid)
    Si_grid = S1
    Si_desvec = Si_grid.flatten() #[(nonpmlNx-Mx)//2:(nonpmlNx-Mx)//2+Mx,(nonpmlNy-My)//2:(nonpmlNy-My)//2+My].flatten()

    O_lin = ((Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dl**2).flatten() 

    #O_quad = (-zinv * dl**2 * (Gddinv.conj().T @ Gddinv)).tocsc()
    dualval0 = -10
    tt = 0
    try:
        len_sp = np.loadtxt(len_sp_name)
        len_sp = int(len_sp)
        subPlist = []
        for ll in range(len_sp):
            spl = sp.load_npz(sp_root_name+str(ll)+'.npz')
            subPlist += [spl]
        Lags = np.loadtxt(lags_name)
    except:
        pass
    while True:
#    for qq in range(10):    
        try:
            print('len(subPlist)', len(subPlist))
            print('Lags', Lags)
            #            plot_sparseP_distribution(Mx, My, np.ones((Mx,My), dtype=np.bool), subPlist, name=name+'_'+str(len(subPlist))+'.png')
            Plist = [Id]
            Plist.extend(subPlist)
            include = [True,True] * len(Plist)
            include[2] = False; include[3] = False #take out one subregion to avoid linear dependence with global constraints
            optLags, optgrad, dualval, objval, violation_r, violation_im = get_Msparse_bound(chi, Si_desvec, O_lin, O_quad, Gddinv, Plist, include, initLags=Lags, getViolation=True, fakeSratio=fakeSratio, iter_period=iter_period)
            
            print('check violation correctness',flush=True)
            for i in range(len(Plist)):
                Recstrt_r = np.sum(Plist[i] @ violation_r)
                Recstrt_im = np.sum(Plist[i] @ violation_im)
                print('optgrad Recstrt_r', optgrad[2*i])
                print('violation Recstrt_r', Recstrt_r)
                print('optgrad Recstrt_im', optgrad[2*i+1])
                print('violation Recstrt_im', Recstrt_im)

            sigma_abs = k0 * dualval
            sigma_enh = sigma_abs / design_y
            print('len(subPlist)', len(subPlist))
            print('sigma_abs', sigma_abs)
            print('sigma_enh', sigma_enh,flush=True)
            if np.absolute((dualval-dualval0)/dualval) < 1e-2:
                break
            else:
                dualval0 = dualval

            if tt % 2 == 0:
                new_subLags, subPlist = split_sparseP_by_violation(optLags[2:], subPlist, violation_r)
                new_Lags = np.zeros(2+len(new_subLags))
                new_Lags[:2] = optLags[:2]
                new_Lags[2:] = new_subLags[:]
                Lags = new_Lags
            else:
                new_subLags, subPlist = split_sparseP_by_violation(optLags[2:], subPlist, violation_im)
                new_Lags = np.zeros(2+len(new_subLags))
                new_Lags[:2] = optLags[:2]
                new_Lags[2:] = new_subLags[:]
                Lags = new_Lags
            tt += 1
                
            
        except ValueError:
            break
    len_sp = len(subPlist)    
    len_sp_file = open(len_sp_name,"w")
    len_sp_file.write(str(len_sp))
    len_sp_file.close()
    for ll in range(len_sp):
        sp.save_npz(sp_root_name+str(ll)+'.npz',subPlist[ll])
    lags_file = open(lags_name,"w")
    np.savetxt(lags_file,Lags)
    lags_file.close()
    Prad_enh = dualval / vacPrad
    return Prad_enh
#    return optLags, optgrad, dualval, objval, sigma_abs, sigma_enh



