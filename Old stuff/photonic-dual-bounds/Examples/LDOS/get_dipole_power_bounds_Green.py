import numpy as np
import scipy.linalg as la

from dualbound.Rect.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_TM_G_od, get_Gddinv

from dualbound.Lagrangian.spatialProjopt_multiSource_Zops_singlematrix_numpy import get_gradZTT, get_gradZTS_S, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity, get_inc_ZTT_mineig

from dualbound.Lagrangian.spatialProjopt_multiSource_dualgradHess_fakeS_singlematrix_numpy import get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix

from dualbound.Lagrangian.spatialProjopt_multiSource_feasiblept_singlematrix import spatialProjopt_find_feasiblept

from dualbound.Optimization.BFGS_fakeSource import BFGS_fakeS


def get_TM_dipole_Prad_bound(chi, wvlgth, design_x, design_y, vacuum_x, vacuum_y, dist, pml_sep, pml_thick, gpr, NProjx, NProjy, opttol=1e-2, fakeSratio=1e-2, iter_period=1000):

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
    halfx = Mx//2
    halfy = My//2
    design_area = Mx*My-vacuumx*vacuumy
    ###############get Green's function for design region#############
    design_mask = np.zeros((nonpmlNx,nonpmlNy), dtype=np.bool)
    design_mask[Npmlsep+Dx:Npmlsep+Dx+Mx , Npmlsep:Npmlsep+My] = True
    design_mask[Npmlsep+Dx+halfx-vacuumx//2:Npmlsep+Dx+halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)])), Npmlsep+halfy-vacuumy//2:Npmlsep+halfy+int(max([vacuumy//2,np.ceil(vacuumy/2)]))] = False

    Gdd = get_TM_G_od(wvlgth, dl, nonpmlNx, nonpmlNy, Npml, design_mask, design_mask)

    U = ((1.0/chi)*np.eye(design_area) - Gdd).conj()
    
    ###############get projection operators#####################
    proj_ulco_list = [(0,0)]
    proj_brco_list = [(Mx,My)]
    subproj_ulco_list, subproj_brco_list = divide_rect_region((0,0), (Mx,My), NProjx, NProjy)
    proj_ulco_list.extend(subproj_ulco_list[1:])
    proj_brco_list.extend(subproj_brco_list[1:]) #include entire design domain, and leave out one subregion projection so constraints are linearly independent
    Pdeslist = []; UPlist = []
    
    len_ulco_list = len(proj_ulco_list)
    for i in range(len(proj_ulco_list)):
         proj_ulco = proj_ulco_list[i]
         proj_brco = proj_brco_list[i]
         Proj_bool = get_rect_proj_in_rect_region((0,0), (Mx,My), proj_ulco, proj_brco)
         Proj_mat = np.zeros_like(Proj_bool, dtype=int)
         Proj_mat[Proj_bool] = 1
         if halfx-vacuumx//2 <= proj_ulco[0] and halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)])) >= proj_brco[0] and halfx-vacuumx//2 <= proj_ulco[1] and halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)])) >= proj_brco[0]:
             len_ulco_list -= 1
             continue
         for k in range(halfy+int(max([vacuumy//2,np.ceil(vacuumy/2)])),halfy-vacuumy//2,-1):
             for i in range(halfx+int(max([vacuumx//2,np.ceil(vacuumx/2)])),halfx-vacuumx//2,-1):
                 Proj_mat = np.delete(Proj_mat,i+Mx*k)
         Proj_mat = np.diag(Proj_mat)
         UP = U @ Proj_mat
         Pdeslist.append(Proj_mat)
         UPlist.append(UP)

    #########################setting up sources#################
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    Ezfield = np.zeros((Nx,Ny), dtype=np.complex)
    vacPrad = 0
    for i in range(halfx,halfx+1):
        for k in range(halfy,halfy+1):
            cx = Npml+Npmlsep+Dx+i
            cy = Npml+Npmlsep+k
            #    cx = Npml + Npmlsep
            #    cy = Ny//2 #position of dipole in entire computational domain
            Ezfield_temp = get_TM_dipole_field(wvlgth, dl, Nx, Ny, cx, cy, Npml) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
            vacPrad_temp = -0.5 * np.real(Ezfield_temp[cx,cy]) #default unit amplitude dipole
            Ezfield += Ezfield_temp
            vacPrad += vacPrad_temp
    S1 = (k0/1j/Z) * (Ezfield[Npml:-Npml,Npml:-Npml])[design_mask] * dl**2 #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_lin = (Z/2/k0) * (1j/2) * S2
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * np.eye(Gdd.shape[0], dtype=np.complex) * dl**2 #same dl**2 factor as above

    gradZTT = get_gradZTT(1, UPlist)
    gradZTS_S = get_gradZTS_S(1, S1, Pdeslist)

    include = [True] * (2*(len_ulco_list)) #include all constraints

    dgfunc = lambda dof, grad, fSl, get_grad=True: get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(1, dof, grad, np.array([]), include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=vacPrad, get_grad=get_grad, get_Hess=False)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(1, dof, include, O_quad, gradZTT)

    mineigfunc = lambda dof: get_inc_ZTT_mineig(1, dof, include, O_quad, gradZTT, eigvals_only=False)

    #find initial point for dual optimization
    Lags = spatialProjopt_find_feasiblept(1, len(include), include, O_quad, gradZTT)
    Lags[1] = np.abs(Lags[1])+0.01
    while check_spatialProj_Lags_validity(1, Lags, O_quad, gradZTT)<=0:
        Lags[1] *= 1.5
        
    #run dual optimization
    optincLags, optincgrad, dualval, objval = BFGS_fakeS(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, iter_period=iter_period)
    
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]
    print('the remaining constraint violations')
    print(optgrad)

    Prad_enh = dualval / vacPrad
    return Prad_enh

