import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys, time

from dualbound.Rect.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

from dualbound.Rect.arb_domains import get_arb_proj_in_rect_region

from dualbound.Maxwell.TM_FDFD import get_TM_dipole_field, get_Gddinv

from get_Msparse_bounds import get_Msparse_bound

from get_dense_bounds import get_multiSource_dense_bound

from dualbound.Rect.rect_iterative_splitting import dualopt_iterative_splitting, dualopt_Msparse_iterative_splitting

from dualbound.Rect.mask_iterative_splitting import dualopt_Msparse_iterative_splitting_general

from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_multiSource_Msparse_gradZTS_S, get_Msparse_gradZTS_S, get_multiSource_Msparse_gradZTT, get_Msparse_gradZTT, get_inc_ZTT_mineig, get_inc_PD_ZTT_mineig, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity

from dualbound.Lagrangian.spatialProjopt_vecs_Msparse import get_ZTTcho_Tvec

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept

from dualbound.Optimization.BFGS_fakeSource import BFGS_fakeS
from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart



def get_TM_dipole_oneside_Prad_Msparse_iterative_splitting(chi, wvlgth, design_x, design_y, vacuum_x, vacuum_y, emitter_x, emitter_y, dist, pml_sep, pml_thick, gprx, gpry, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, alg='Newton'):

    zinv = np.imag(chi) / np.real(chi*np.conj(chi))
    
    domain_x = 2*pml_sep + design_x
    domain_y = 2*pml_sep + design_y

    dx = 1.0/gprx
    dy = 1.0/gpry #allow for different discretizations in x,y direction

    Mx = int(np.round(design_x/dx))
    My = int(np.round(design_y/dy))
    Dx = int(np.round(dist/dx))
    Npmlx = int(np.round(pml_thick/dx))
    Npmly = int(np.round(pml_thick/dy))
    Npmlsepx = int(np.round(pml_sep/dx))
    Npmlsepy = int(np.round(pml_sep/dy))
    nonpmlNx = Mx + Dx + 2*Npmlsepx
    nonpmlNy = My + 2*Npmlsepy
    Nx = nonpmlNx + 2*Npmlx
    Ny = nonpmlNy + 2*Npmly
    emitterx = int(np.round(emitter_x / dx))
    emittery = int(np.round(emitter_y / dy))
    vacuumx = int(np.round(vacuum_x / dx))
    vacuumy = int(np.round(vacuum_y / dy))


    ###############get Green's function for design region#############
    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npmlx+Npmlsepx+Dx:Npmlx+Npmlsepx+Dx+Mx , Npmly+Npmlsepy:Npmly+Npmlsepy+My] = True
    design_mask[Npmlx+Npmlsepx+Dx+(Mx-vacuumx)//2:Npmlx+Npmlsepx+Dx+(Mx-vacuumx)//2+vacuumx,Npmly+Npmlsepy+(My-vacuumy)//2:Npmly+Npmlsepy+(My-vacuumy)//2+vacuumy] = False
    design_mask_proj = np.zeros((Mx,My), dtype=np.bool)
    design_mask_proj[:,:] = True
    design_mask_proj[(Mx-vacuumx)//2:(Mx-vacuumx)//2+vacuumx,(My-vacuumy)//2:(My-vacuumy)//2+vacuumy] = False

    print('getting Gddinv', flush=True)
    Gddinv, _ = get_Gddinv(wvlgth, dx, Nx, Ny, Npmlx, design_mask)

    print('finished computing Gddinv', flush=True)
    
    UM = (Gddinv.conj().T @ Gddinv)/np.conj(chi) - Gddinv

    AsymUM = (UM - UM.conj().T) / 2j
    print('UM format', UM.format, 'AsymUM format', AsymUM.format, flush=True)

    Id = sp.eye(Gddinv.shape[0], format="csc")
    Plist = [Id.copy()]
    Lags = None
    
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    Ezfield = np.zeros((Nx,Ny),dtype=np.complex)
    vacPrad = 0
    for k in range(Ny//2+int(max([emittery//2,np.ceil(emittery/2)])),Ny//2-emittery//2,-1):
        for ii in range(Nx//2+int(max([emitterx//2,np.ceil(emitterx/2)])),Nx//2-emitterx//2,-1):
            cx = ii
            cy = k #position of dipole in entire computational domain

            Ezfield_temp = get_TM_dipole_field(wvlgth, dx, Nx, Ny, cx, cy, Npmlx) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)
            Ezfield += Ezfield_temp
    for k in range(Ny//2+int(max([emittery//2,np.ceil(emittery/2)])),Ny//2-emittery//2,-1):
        for ii in range(Nx//2+int(max([emitterx//2,np.ceil(emitterx/2)])),Nx//2-emitterx//2,-1):
            cx = ii
            cy = k
            vacPrad_temp = -0.5 * np.real(Ezfield[cx,cy]) #default unit amplitude dipole
            vacPrad += vacPrad_temp
    S1 = (k0/1j/Z) * Ezfield[design_mask] #S1 = G @ J; dl**2 factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_quad = (Z/2/k0) * (np.imag(chi)/np.real(chi*np.conj(chi))) * (Gddinv.conj().T @ Gddinv).tocsc() * dx * dy #same dl**2 factor as above

    Si_grid = S1
    Si_desvec = Si_grid.flatten() 

    O_lin = ((Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dx * dy).flatten() 


    print('starting iterative splitting', flush=True)

    def Prad_outputfunc(optLags, optgrad, dualval, objval, vacPrad):
        Prad_enh = dualval / vacPrad
        print('Prad_enh', Prad_enh, flush=True)
    outputFunc = lambda L,G,D,O: Prad_outputfunc(L,G,D,O, vacPrad)

    dualoptFunc = lambda nS, initL, GinvdagPdagl, UPl, inc: get_Msparse_bound(S1, O_lin, O_quad, GinvdagPdagl, UPl, inc, initLags=initL, dualconst=vacPrad, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period, alg=alg)

    dualopt_Msparse_iterative_splitting_general(1, Mx, My, design_mask_proj, chi, Gddinv, dualoptFunc, outputFunc, pol='TM')

    

