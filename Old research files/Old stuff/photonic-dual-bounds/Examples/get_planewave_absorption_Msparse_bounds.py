import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import DivergingNorm
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys, time
sys.path.append('../')

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_mSmF_Msparse_gradZTS_S, get_multiSource_Msparse_gradZTS_S, get_Msparse_gradZTS_S, get_mSmF_Msparse_gradZTT, get_multiSource_Msparse_gradZTT, get_Msparse_gradZTT, get_inc_ZTT_mineig, get_inc_PD_ZTT_mineig, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity

from dualbound.Lagrangian.spatialProjopt_vecs_Msparse import get_ZTTcho_Tvec

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept

from dualbound.Optimization.BFGS_fakeSource import BFGS_fakeS
from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart


from dualbound.Maxwell.TM_FDFD import get_Gddinv
from dualbound.Constraints.rect_domains import get_rect_proj_in_rect_region, divide_rect_region
from dualbound.Constraints.rect_iterative_splitting import dualopt_Msparse_iterative_splitting
from dualbound.Constraints.mask_iterative_splitting import split_sparseP_by_violation, plot_sparseP_distribution, get_max_violation_sparseP


def get_mSmF_Msparse_bound(chilist, Si_st, O_lin_st, O_quad_st, Ginvlist, Plist, include, dualconst=0.0, initLags=None, opttol=1e-2, fakeSratio=1e-2, iter_period=20):

    n_basis = Plist[0].shape[0]
    n_S = len(Si_st) // n_basis

    gradZTT = get_mSmF_Msparse_gradZTT(n_S, chilist, Ginvlist, Plist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad_st, gradZTT)
    gradZTS_S = get_mSmF_Msparse_gradZTS_S(n_S, Si_st, Ginvlist, Plist)

    dgfunc = lambda dof, dofgrad, fSl, get_grad=True: get_inc_spatialProj_dualgrad_fakeS_Msparse(dof, dofgrad, include, O_lin_st, O_quad_st, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, chofac=ZTTchofac)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(dof, include, O_quad_st, gradZTT, chofac=ZTTchofac)

    mineigfunc = lambda dof: get_inc_PD_ZTT_mineig(dof, include, O_quad_st, gradZTT, eigvals_only=False)

    if initLags is None:
        Lags = spatialProjopt_find_feasiblept(len(include), include, O_quad_st, gradZTT)
        #Lags = np.random.rand(len(include))
    else:
        Lags = initLags.copy()

    print('Lags', Lags)
    zeta_mask = np.zeros_like(Lags, dtype=np.bool)
    n_cplx_projLags = n_S**2
    for i in range(n_S):
        zeta_mask[n_cplx_projLags + i*n_S + i] = True
        Lags[zeta_mask] = np.abs(Lags[zeta_mask]) + 0.01
        
    while True:
        tmp = check_spatialProj_Lags_validity(Lags, O_quad_st, gradZTT)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[zeta_mask] *= 1.5

    #optincLags, optincgrad, dualval, objval = BFGS_fakeS(Lags[include], Si, dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    
    
    optLags = np.zeros(len(include), dtype=np.double)
    optLags[include] = optincLags[:]
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]

    print('the remaining constraint violations')
    print(optgrad)

    return optLags, optgrad, dualval, objval


def get_Msparse_1designrect_normal_absorption(chi, wvlgth, design_x, design_y, pml_sep, pml_thick, gpr, NProjx, NProjy, initLags=None, fakeSratio=1e-2, iter_period=20):

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
    xgrid = dl * np.arange(nonpmlNx)
    ygrid = dl * np.arange(nonpmlNy)
    print('xgrid sep', xgrid[1]-xgrid[0], 'dl', dl)
    xgrid,ygrid = np.meshgrid(xgrid,ygrid, indexing='ij')
    k0 = 2*np.pi/wvlgth
    phasegrid = k0*xgrid #normal incidence
    Si_grid = np.exp(1j*phasegrid)
    Si_desvec = Si_grid[(nonpmlNx-Mx)//2:(nonpmlNx-Mx)//2+Mx,(nonpmlNy-My)//2:(nonpmlNy-My)//2+My].flatten()
        
    O_lin = np.zeros_like(Si_desvec, dtype=np.complex)

    O_quad = (-zinv * dl**2 * (Gddinv.conj().T @ Gddinv)).tocsc()


    optLags, optgrad, dualval, objval = get_mSmF_Msparse_bound([chi], Si_desvec, O_lin, O_quad, [Gddinv], Plist, include, fakeSratio=fakeSratio, iter_period=iter_period)
    
    sigma_abs = k0 * dualval
    sigma_enh = sigma_abs / design_y
    return optLags, optgrad, dualval, objval, sigma_abs, sigma_enh


def get_multifreq_1designrect_normal_absorption(chilist, wvlgth_list, design_x, design_y, pml_sep_list, pml_thick_list, res, NProjx, NProjy, incCross=True, maxminlist=None, initLags=None, fakeSratio=1e-2, iter_period=20):
    """
    get bounds on summed planewave absorption cross section for multiple frequencies
    since we want to use cross-source constraints the grid size is fixed for different frequencies; res is # of gridpts per unit length (not wavelength!)
    however we can have different pml thickness and separation depending on wavelength
    maxminlist[i] is positive if we want to maximize absorption at wvlgth[i] negative if we want to minimize; default all maximize
    """
    if maxminlist is None:
        maxminlist = np.ones(len(chilist))

    Mx = int(np.round(design_x * res))
    My = int(np.round(design_y * res))


    ###############get Plist####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]

    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)

    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:])
    
    Plist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        Plist.append(Proj_mat)

    #############get Si_st and Ginvlist#################
    dl = 1.0 / res #resolution defined relative to unit length, not necessarily wavelength
    n_basis = Mx*My
    n_S = len(chilist)
    n_pdof = n_S * n_basis

    Ginvlist = []
    Si_st = np.zeros(n_S*n_basis, dtype=np.complex)
    O_quad_st = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)

    for i in range(n_S):
        chi = chilist[i]
        zinv = np.imag(chi) / np.real(chi*np.conj(chi))
        wvlgth = wvlgth_list[i]
        pml_sep = pml_sep_list[i]
        pml_thick = pml_thick_list[i]

        Npml = int(np.round(pml_thick * res))
        Npml_sep = int(np.round(pml_sep * res))
        Nx = 2*(Npml+Npml_sep) + Mx
        Ny = 2*(Npml+Npml_sep) + My

        designMask = np.zeros((Nx,Ny), dtype=np.bool)
        designMask[Npml+Npml_sep:-(Npml+Npml_sep),Npml+Npml_sep:-(Npml+Npml_sep)] = True

        Gddinv, _ = get_Gddinv(wvlgth, dl, Nx, Ny, (Npml,Npml), designMask)
        Ginvlist.append(Gddinv)
        
        """
        ###setting up normal incident planewave source###
        xgrid = np.linspace(-Mx*dl/2.0, Mx*dl/2.0, Mx)
        ygrid = np.linspace(-My*dl/2.0, My*dl/2.0, My)
        """
        
        ###setting up normal incident planewave source###
        xgrid = dl * np.arange(Mx)
        ygrid = dl * np.arange(My)

        print('xgrid sep', xgrid[1]-xgrid[0], 'dl', dl)
        xgrid,ygrid = np.meshgrid(xgrid,ygrid, indexing='ij')
        k0 = 2*np.pi/wvlgth
        phasegrid = k0*xgrid #normal incidence
        Si_grid = np.exp(1j*phasegrid)
        Si_desvec = Si_grid.flatten()
        Si_st[i*n_basis:(i+1)*n_basis] = Si_desvec
        
        O_quad = -zinv * k0 * dl**2 * maxminlist[i] * (Gddinv.conj().T @ Gddinv)
        O_quad_st[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis] = O_quad

    O_quad_st = O_quad_st.tocsc()
    O_lin_st = np.zeros_like(Si_st, dtype=np.complex)

    if incCross:
        print('include everything')
        include = [True] * (2*len(Plist)*n_S**2) #include everything
    else:
        print('include indep')
        include = [False] * (2*len(Plist)*n_S**2) #simultaneously solve two uncoupled problems
        for l in range(len(Plist)):
            for i in range(n_S):
                include[2*l*n_S**2 + i*n_S + i] = True
                include[(2*l+1)*n_S**2 + i*n_S + i] = True
    

    optLags, optgrad, dualval, objval = get_mSmF_Msparse_bound(chilist, Si_st, O_lin_st, O_quad_st, Ginvlist, Plist, include, fakeSratio=fakeSratio, iter_period=iter_period)
    
    sigma_abs_tot = dualval
    sigma_abs_avg = sigma_abs_tot / n_S
    sigma_enh_avg = sigma_abs_avg / design_y

    """
    #test singles
    include0 = [True,True] * len(Plist)
    print('chi', chilist[0], 'Ginv shape', Ginvlist[0].shape)
    optLags, optgrad, dualval, objval = get_mSmF_Msparse_bound([chilist[0]], Si_st[:n_basis], O_lin_st[:n_basis], O_quad_st[:n_basis,:n_basis], [Ginvlist[0]], Plist, include0, fakeSratio=fakeSratio, iter_period=iter_period)
    
    sigma_abs_1 = dualval
    sigma_enh_1 = sigma_abs_1 / design_y
    

    include1 = [True,True] * len(Plist)
    print('chi', chilist[1], 'Ginv shape', Ginvlist[1].shape)
    optLags, optgrad, dualval, objval = get_mSmF_Msparse_bound([chilist[1]], Si_st[n_basis:2*n_basis], O_lin_st[n_basis:2*n_basis], O_quad_st[n_basis:2*n_basis,n_basis:2*n_basis], [Ginvlist[1]], Plist, include1, fakeSratio=fakeSratio, iter_period=iter_period)
    
    sigma_abs_2 = dualval
    sigma_enh_2 = sigma_abs_2 / design_y

    print('test first source on its own,', 'sigma_abs', sigma_abs_1, 'sigma_enh', sigma_enh_1)
    print('test second source on its own,', 'sigma_abs', sigma_abs_2, 'sigma_enh', sigma_enh_2)
    """
    
    return optLags, optgrad, dualval, objval, sigma_abs_tot, sigma_abs_avg, sigma_enh_avg



def get_periodicY_multifreq_1designrect_normal_absorption(chilist, wvlgth_list, design_x, design_y, domain_y, pml_sep_list, pml_thick_list, res, NProjx, NProjy, incCross=True, maxminlist=None, initLags=None, fakeSratio=1e-2, iter_period=20):
    """
    get bounds on summed planewave absorption cross section for multiple frequencies
    since we want to use cross-source constraints the grid size is fixed for different frequencies; res is # of gridpts per unit length (not wavelength!)
    however we can have different pml thickness and separation depending on wavelength
    maxminlist[i] is positive if we want to maximize absorption at wvlgth[i] negative if we want to minimize; default all maximize
    """
    if maxminlist is None:
        maxminlist = np.ones(len(chilist))

    Mx = int(np.round(design_x * res))
    My = int(np.round(design_y * res))


    ###############get Plist####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]

    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)

    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:])
    
    Plist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        Plist.append(Proj_mat)

    #############get Si_st and Ginvlist#################
    dl = 1.0 / res #resolution defined relative to unit length, not necessarily wavelength
    n_basis = Mx*My
    n_S = len(chilist)
    n_pdof = n_S * n_basis

    Ginvlist = []
    Si_st = np.zeros(n_S*n_basis, dtype=np.complex)
    O_quad_st = sp.lil_matrix((n_pdof,n_pdof), dtype=np.complex)

    for i in range(n_S):
        chi = chilist[i]
        zinv = np.imag(chi) / np.real(chi*np.conj(chi))
        wvlgth = wvlgth_list[i]
        pml_sep = pml_sep_list[i]
        pml_thick = pml_thick_list[i]

        Npml = int(np.round(pml_thick * res))
        Npml_sep = int(np.round(pml_sep * res))
        Nx = 2*(Npml+Npml_sep) + Mx
        Ny = int(np.round(domain_y / dl))

        designMask = np.zeros((Nx,Ny), dtype=np.bool)
        designMask[Npml+Npml_sep:-(Npml+Npml_sep),(Ny-My)//2:(Ny-My)//2+My] = True

        print('Nx', Nx, 'Ny', Ny, 'Mx', Mx, 'My', My, 'Npml', Npml, 'shape designMask', designMask.shape)
        Gddinv, _ = get_Gddinv(wvlgth, dl, Nx, Ny, (Npml,0), designMask) #periodic in Y direction
        Ginvlist.append(Gddinv)
        
        ###setting up normal incident planewave source###
        xgrid = dl * np.arange(Mx)
        ygrid = dl * np.arange(My)

        print('xgrid sep', xgrid[1]-xgrid[0], 'dl', dl)
        xgrid,ygrid = np.meshgrid(xgrid,ygrid, indexing='ij')
        k0 = 2*np.pi/wvlgth
        phasegrid = k0*xgrid #normal incidence
        Si_grid = np.exp(1j*phasegrid)
        Si_desvec = Si_grid.flatten()
        Si_st[i*n_basis:(i+1)*n_basis] = Si_desvec
        
        O_quad = -zinv * k0 * dl**2 * maxminlist[i] * (Gddinv.conj().T @ Gddinv)
        O_quad_st[i*n_basis:(i+1)*n_basis,i*n_basis:(i+1)*n_basis] = O_quad

    O_quad_st = O_quad_st.tocsc()
    O_lin_st = np.zeros_like(Si_st, dtype=np.complex)

    if incCross:
        print('include everything')
        include = [True] * (2*len(Plist)*n_S**2) #include everything
    else:
        print('include indep')
        include = [False] * (2*len(Plist)*n_S**2) #simultaneously solve two uncoupled problems
        for l in range(len(Plist)):
            for i in range(n_S):
                include[2*l*n_S**2 + i*n_S + i] = True
                include[(2*l+1)*n_S**2 + i*n_S + i] = True
    

    optLags, optgrad, dualval, objval = get_mSmF_Msparse_bound(chilist, Si_st, O_lin_st, O_quad_st, Ginvlist, Plist, include, fakeSratio=fakeSratio, iter_period=iter_period)
    
    sigma_abs_tot = dualval
    sigma_abs_avg = sigma_abs_tot / n_S
    sigma_enh_avg = sigma_abs_avg / design_y
    
    return optLags, optgrad, dualval, objval, sigma_abs_tot, sigma_abs_avg, sigma_enh_avg
