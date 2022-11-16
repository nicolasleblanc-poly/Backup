import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys
sys.path.append('../')

#methods to generate projection operators over sub-regions
from dualbound.Constraints.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

#methods that compute the relevant physical information
from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_Gddinv

from dualbound.Maxwell.Yee_TE_FDFD import get_TE_dipole_field, get_Yee_TE_Gddinv

#methods for forming the Lagrangian, evaluating the dual value, and determining domain of duality
from dualbound.Lagrangian.spatialProjopt_vecs_Msparse import get_Tvec, get_ZTTcho_Tvec

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_Msparse_gradZTS_S, get_Msparse_gradZTT, get_inc_PD_ZTT_mineig, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity

#optimization method to compute dual optimum
from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart


def get_TM_dipole_Prad_Msparse_bound(chi, wvlgth, design_x, design_y, dist, pml_sep, pml_thick, gpr, NProjx, NProjy, opttol=1e-2, fakeSratio=1e-2, iter_period=20):

    """
    Evaluates limit of enhancement to dipole radiation power in 2D for a TM dipole
    some distance away from a rectangular design region. 
    
    Parameters
    ----------
    chi : complex
        susceptibility of the design material
    wvlgth : real
        the wavelength of light, in units of 1
    design_x : real
        x-dimension of rectangular design region, in units of 1
    design_y : real
        y-dimension of rectangular design region, in units of 1
    dist : real
        separation between dipole and near edge of design region along x-direction, in units of 1
    pml_sep : real
        seperation between pml boundary and nearest computational feature, in units of 1
    pml_thick : real
        thickness of pml region, in units of 1
    gpr : integer
        number of computational pixels per unit of 1
    NProjx : integer
        number of subregion cuts along the x direction
    NProjy : integer
        number of subregion cuts along the y direction
    opttol : real, optional
        The relative optimization convergence tolerance. The default is 1e-2.
    fakeSratio : real, optional
        The relative size of added fake source terms when doing dual optimization. The default is 1e-2.
    iter_period : integer, optional
        Number of iterations before optimization checks again whether it is getting stuck. The default is 20.

    Returns
    -------
    Prad_enh : real
        Limit on the enhancement to dipole radiation power

    """
    
    dl = 1.0/gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Dx = int(np.round(dist/dl))
    Npml = int(np.round(pml_thick/dl))
    Npmlsep = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + Dx + 2*Npmlsep
    nonpmlNy = My + 2*Npmlsep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    ###############get Green's function for design region#############
    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npml+Npmlsep+Dx:Npml+Npmlsep+Dx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True

    Gddinv, _ = get_Gddinv(wvlgth, dl, Nx, Ny, Npml, design_mask)

    UM = (Gddinv.conj().T @ Gddinv)/np.conj(chi) - Gddinv
    
    ###############get projection operators#####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]
    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)
    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:]) #include entire design domain, and leave out one subregion projection so constraints are linearly independent
    
    GinvdagPdaglist = []
    UPlist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        GinvdagPdag = Gddinv.conj().T @ Proj_mat
        UMP = (Gddinv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
        print('GinvdagPdag format', GinvdagPdag.format)
        print('UMP format', UMP.format, flush=True)
        GinvdagPdaglist.append(GinvdagPdag.tocsc())
        UPlist.append(UMP.tocsc())
    
    #########################setting up sources#################
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    cx = Npml + Npmlsep
    cy = Ny//2 #position of dipole in entire computational domain
    Ezfield = get_TM_dipole_field(wvlgth, dl, Nx, Ny, cx, cy, Npml) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
    vacPrad = -0.5 * np.real(Ezfield[cx,cy]) #default unit amplitude dipole
    S1 = (k0/1j/Z) * Ezfield[design_mask] * dl**2 #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_lin = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2)
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dl**2 #same dl**2 factor as above

    
    gradZTT = get_Msparse_gradZTT(UPlist)
    gradZTS_S = get_Msparse_gradZTS_S(S1, GinvdagPdaglist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)

    include = [True] * (2*len(proj_ulco_list)) #include all constraints

    dgfunc = lambda dof, grad, fSl, get_grad=True: get_inc_spatialProj_dualgrad_fakeS_Msparse(dof, grad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=vacPrad, get_grad=get_grad, chofac=ZTTchofac)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)

    mineigfunc = lambda dof: get_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    #find initial point for dual optimization
    #Lags = spatialProjopt_find_feasiblept(len(include), include, O_quad, gradZTT)
    Lags = np.random.rand(len(include))
    Lags[1] = np.abs(Lags[1])+0.01
    while check_spatialProj_Lags_validity(Lags, O_quad, gradZTT)<=0:
        Lags[1] *= 1.5
        
    #run dual optimization
    optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]
    print('the remaining constraint violations')
    print(optgrad)

    Prad_enh = dualval / vacPrad
    return Prad_enh


def get_TE_dipole_Prad_Msparse_bound(chi, wvlgth, pol, design_x, design_y, dist, pml_sep, pml_thick, gpr, NProjx, NProjy, opttol=1e-2, fakeSratio=1e-2, iter_period=20, randomLags=False, randomGddT=False, returnAll=False):
    
    """
    Evaluates limit of enhancement to dipole radiation power in 2D for a TE dipole
    some distance away from a rectangular design region. 
    
    returnAll, returnRandom specified for testing divergence free criterion satisfied by optimal fields
    Parameters
    ----------
    chi : complex
        susceptibility of the design material
    wvlgth : real
        the wavelength of light, in units of 1
    design_x : real
        x-dimension of rectangular design region, in units of 1
    design_y : real
        y-dimension of rectangular design region, in units of 1
    dist : real
        separation between dipole and near edge of design region along x-direction, in units of 1
    pml_sep : real
        seperation between pml boundary and nearest computational feature, in units of 1
    pml_thick : real
        thickness of pml region, in units of 1
    gpr : integer
        number of computational pixels per unit of 1
    NProjx : integer
        number of subregion cuts along the x direction
    NProjy : integer
        number of subregion cuts along the y direction
    opttol : real, optional
        The relative optimization convergence tolerance. The default is 1e-2.
    fakeSratio : real, optional
        The relative size of added fake source terms when doing dual optimization. The default is 1e-2.
    iter_period : integer, optional
        Number of iterations before optimization checks again whether it is getting stuck. The default is 20.

    Returns
    -------
    Prad_enh : real
        Limit on the enhancement to dipole radiation power

    """

    dl = 1.0/gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Dx = int(np.round(dist/dl))
    Npml = int(np.round(pml_thick/dl))
    Npmlsep = int(np.round(pml_sep/dl))
    nonpmlNx = Mx + Dx + 2*Npmlsep
    nonpmlNy = My + 2*Npmlsep
    Nx = nonpmlNx + 2*Npml
    Ny = nonpmlNy + 2*Npml

    ###############get Green's function for design region#############
    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npml+Npmlsep+Dx:Npml+Npmlsep+Dx+Mx , Npml+Npmlsep:Npml+Npmlsep+My] = True

    Gddinv, _ = get_Yee_TE_Gddinv(wvlgth, dl, dl, Nx, Ny, Npml, Npml, design_mask)

    UM = (Gddinv.conj().T @ Gddinv)/np.conj(chi) - Gddinv
    
    ###############get projection operators#####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]
    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)
    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:]) #include entire design domain, and leave out one subregion projection so constraints are linearly independent
    
    GinvdagPdaglist = []
    UPlist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.kron(sp.diags(Proj_mat), sp.eye(2), format="csc")
        GinvdagPdag = Gddinv.conj().T @ Proj_mat
        UMP = (Gddinv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
        print('GinvdagPdag format', GinvdagPdag.format)
        print('UMP format', UMP.format, flush=True)
        GinvdagPdaglist.append(GinvdagPdag.tocsc())
        UPlist.append(UMP.tocsc())
    
    #########################setting up sources#################
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    cx = Npml + Npmlsep
    cy = Ny//2 #position of dipole in entire computational domain

    if pol=="x":
        _, Exfield, Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npml, Npml, dl ,dl, cx, cy, 1, amp=1.0/dl**2)
        vacPrad = -0.5 * np.real(Exfield[cx,cy]) #default unit amplitude dipole
    else:
        _, Exfield, Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npml, Npml, dl ,dl, cx, cy, 2, amp=1.0/dl**2)
        vacPrad = -0.5 * np.real(Eyfield[cx,cy]) #default unit amplitude dipole

    
    n_basis = int(np.sum(design_mask)) * 2
    S1 = np.zeros(n_basis, dtype=np.complex)
    S1[::2] = Exfield[design_mask]
    S1[1::2] = Eyfield[design_mask]
    S1 *= (k0/1j/Z) * dl**2 #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_lin = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2)
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dl**2 #same dl**2 factor as above

    
    gradZTT = get_Msparse_gradZTT(UPlist)
    gradZTS_S = get_Msparse_gradZTS_S(S1, GinvdagPdaglist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)

    include = [True] * (2*len(proj_ulco_list)) #include all constraints

    dgfunc = lambda dof, grad, fSl, get_grad=True: get_inc_spatialProj_dualgrad_fakeS_Msparse(dof, grad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=vacPrad, get_grad=get_grad, chofac=ZTTchofac)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)

    mineigfunc = lambda dof: get_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    #find initial point for dual optimization
    #Lags = spatialProjopt_find_feasiblept(len(include), include, O_quad, gradZTT)
    Lags = np.random.rand(len(include))
    Lags[1] = np.abs(Lags[1])+0.01
    while check_spatialProj_Lags_validity(Lags, O_quad, gradZTT)<=0:
        Lags[1] *= 1.5
        
    #run dual optimization
    if not randomLags:
        optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)

        optgrad = np.zeros(len(include), dtype=np.double)
        optgrad[include] = optincgrad[:]
        print('the remaining constraint violations')
        print(optgrad)

        Prad_enh = dualval / vacPrad
    else:
        optincLags = np.random.rand(int(np.sum(include)))
        Prad_enh = 'testing random Lags for div free condition'
        
    if returnAll:
        optLags = np.zeros(len(include), dtype=np.double)
        optLags[include] = optincLags
        ZTS_S = O_lin.copy()
        ZTT = O_quad.copy()
        for i in range(len(optLags)):
            ZTS_S += optLags[i] * gradZTS_S[i]
            ZTT += optLags[i] * gradZTT[i]

        #_, opt_Gdd_T = get_ZTTcho_Tvec(ZTT, ZTS_S, ZTTchofac)
        if randomGddT:
            opt_Gdd_T = np.random.rand(2*Mx*My)-0.5 + 1j*(np.random.rand(2*Mx*My)-0.5)
        elif randomLags:
            opt_Gdd_T = get_Tvec(ZTT, ZTS_S)
        else:
            _, opt_Gdd_T = get_ZTTcho_Tvec(ZTT, ZTS_S, ZTTchofac)
        opt_Gdd_Tx = np.reshape(opt_Gdd_T[::2], (Mx,My))
        opt_Gdd_Ty = np.reshape(opt_Gdd_T[1::2], (Mx,My))
        Etotxfield = np.reshape(Exfield[design_mask], (Mx,My)) + opt_Gdd_Tx
        Etotyfield = np.reshape(Eyfield[design_mask], (Mx,My)) + opt_Gdd_Ty
        
        optT = Gddinv @ opt_Gdd_T
        optTx = np.reshape(optT[::2], (Mx,My))
        optTy = np.reshape(optT[1::2], (Mx,My))
        
        return Prad_enh, design_mask, Exfield, Eyfield, Etotxfield, Etotyfield, optTx, optTy
    else:
        return Prad_enh
