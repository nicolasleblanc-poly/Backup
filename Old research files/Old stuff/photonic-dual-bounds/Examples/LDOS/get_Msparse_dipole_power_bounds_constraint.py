import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys

from dualbound.Rect.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_Gddinv

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_Msparse_gradZTS_S, get_Msparse_gradZTT, get_inc_PD_ZTT_mineig, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity

from dualbound.Optimization.BFGS_fakeSource import BFGS_fakeS


def get_TM_dipole_Prad_Msparse_bound(chi, wvlgth, design_x, design_y, vacuum_x, vacuum_y, dist, pml_sep, pml_thick, gpr, NProjx, NProjy, opttol=1e-2, fakeSratio=1e-2, iter_period=20):

    dl = 1.0/gpr
    Mx = int(np.round(design_x/dl))
    My = int(np.round(design_y/dl))
    Dx = int(np.round(dist/dl))
    vacuumx = int(np.round(vacuum_x/dl))
    vacuumy = int(np.round(vacuum_y/dl))
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
    Upro = np.zeros((Mx*My,Mx*My))
    halfx = Mx//2
    halfy = My//2
    vacuum_region = []
    for i in range(halfx-vacuumx//2,halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)]))):
        for k in range(halfy-vacuumy//2,halfy+int(max([vacuumy//2,np.ceil(vacuumy/2)]))):
            Upro[i+Mx*k,i+Mx*k] = 1.0
    Upro = sp.csc_matrix(Upro)
    Upro = Gddinv.conj().T @ Upro @ Gddinv
    ###############get projection operators#####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]
    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)
    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:]) #include entire design domain, and leave out one subregion projection so constraints are linearly independent
    
    GinvconjPlist = []
    UPlist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = sp.diags(Proj_mat, format="csc")
        GinvconjP = Gddinv.conj().T @ Proj_mat
        UMP = (Gddinv.conj().T @ GinvconjP.conj().T)/np.conj(chi) - GinvconjP.conj().T
        print('GinvconjP format', GinvconjP.format)
        print('UMP format', UMP.format, flush=True)
        GinvconjPlist.append(GinvconjP.tocsc())
        UPlist.append(UMP.tocsc())

    UP = Upro.tocsc()
    Proj_mat_temp = Gddinv.conj().T @ (0*Proj_mat)
    GinvconjPlist.append(Proj_mat_temp.tocsc())
    UPlist.append(UP)
    
    #########################setting up sources#################
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    Ezfield = np.zeros((Nx,Ny), dtype=np.complex)
    vacPrad = 0
    for i in range(halfx-vacuumx//2,halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)]))):
        for k in range(halfy-vacuumy//2,halfy+int(max([vacuumy//2,np.ceil(vacuumy/2)]))):
            cx = Npml+Npmlsep+Dx+i
            cy = Npml+Npmlsep+k
            #    cx = Npml + Npmlsep
            #    cy = Ny//2 #position of dipole in entire computational domain
            Ezfield_temp = get_TM_dipole_field(wvlgth, dl, Nx, Ny, cx, cy, Npml) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
            vacPrad_temp = -0.5 * np.real(Ezfield_temp[cx,cy]) #default unit amplitude dipole
            Ezfield += Ezfield_temp
            vacPrad += vacPrad_temp
    S1 = (k0/1j/Z) * Ezfield[design_mask] * dl**2 #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_lin = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2)
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dl**2 #same dl**2 factor as above

    
    gradZTT = get_Msparse_gradZTT(UPlist)
    gradZTS_S = get_Msparse_gradZTS_S(S1, GinvconjPlist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)

    include = [True] * (2*(len(proj_ulco_list)+1)) #include all constraints

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
    optincLags, optincgrad, dualval, objval = BFGS_fakeS(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]
    print('the remaining constraint violations')
    print(optgrad)

    Prad_enh = dualval / vacPrad
    return Prad_enh
