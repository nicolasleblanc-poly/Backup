import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import time,sys
sys.path.append('C:/Users/Nic.Leblanc/Documents/GitHub/Extracted-power-optimization/photonic-dual-bounds') #load package directory into system path so the script can access the package

from dualbound.Constraints.rect_domains import get_rect_proj_in_rect_region, divide_rect_region

from dualbound.Maxwell.Yee_TE_FDFD import get_TE_dipole_field, get_Yee_TE_Gddinv, get_Yee_TE_GreenFcn, get_Yee_TE_div_mask

from get_Msparse_bounds import get_Msparse_bound

from get_dense_bounds import get_multiSource_dense_bound

def get_TE_dipole_oneside_LDOS_Msparse_bound(chi, wvlgth, orient, design_x, design_y, dist, pml_sep, pml_thick, gprx, gpry, NProjx, NProjy, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, alg='Newton'):
    """
    computes the dual bound for LDOS enhancement given a TE (in-plane) dipole source some distance
    from a rectangular design region

    chi: np.complex
    The material susceptibility
    wvlgth: np.double
    The wavelength of interest, in units of 1.
    orient: 1x2 np.array of np.double
    direction in which the dipole is oriented
    design_x: np.double
    dimension of the design region along the x-direction, in units of 1.
    design_y: np.double
    dimension of the design region along the y-direction, in units of 1.
    dist: np.double
    distance between the dipole and the design region along the x direction, in units of 1
    pml_sep: np.double
    distance between the dipole+structure system and the beginning of the surround PML layers, in units of 1
    pml_thick: np.double
    thickness of surrounding PML layers, in units of 1
    gprx: int
    number of pixels per unit 1 length along the x-direction
    gpry: int
    number of pixels per unit 1 length along the y-direction
    NProjx: int
    Number of projection regions for constraints along the x direction
    Nprojy: int
    Number of projection regions for constraints along the y direction
    -------------------------------------]
    Returns
    LDOS_enh: np.double
    ratio of the calculated bound on LDOS and the vacuum LDOS
    """
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
    ###############get Green's function for design region#############
    design_mask = np.zeros((Nx,Ny), dtype=np.bool)
    design_mask[Npmlx+Npmlsepx+Dx:Npmlx+Npmlsepx+Dx+Mx , Npmly+Npmlsepy:Npmly+Npmlsepy+My] = True
    print('getting Gddinv', flush=True)
    Gddinv, _ = get_Yee_TE_Gddinv(wvlgth, dx, dy, Nx, Ny, Npmlx, Npmly, design_mask, ordering='pol')
    print('finished computing Gddinv', flush=True)
    
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
        Proj_mat = sp.kron(sp.eye(2), sp.diags(Proj_mat), format='csc')
        GinvdagPdag = Gddinv.conj().T @ Proj_mat
        UMP = (Gddinv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T
        
        GinvdagPdaglist.append(GinvdagPdag.tocsc())
        UPlist.append(UMP.tocsc())

    #########################setting up sources#################
    k0 = 2*np.pi / wvlgth
    Z = 1.0 #dimensionless units
    cx = Npmlx + Npmlsepx
    cy = Ny//2 #position of dipole in entire computational domain

    orient / la.norm(orient) #normalize orient so that it is a unit vector
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, amp=1.0/dx/dy) #note that the default amplitude of the dipole source is 1 (scaled by pixel size)

    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, amp=1.0/dx/dy)

    orient_Exfield = orient[0] * x_Exfield + orient[1] * y_Exfield
    orient_Eyfield = orient[0] * x_Eyfield + orient[1] * y_Eyfield

    vacPrad = -0.5 * np.real(orient[0] * orient_Exfield[cx,cy] + orient[1] * orient_Eyfield[cx,cy])
    
    #S1 = (k0/1j/Z) * np.concatenate((Exfield[design_mask], Eyfield[design_mask])) * dx * dy #S1 = G @ J; dx*dy factor to account for integration over pixels as represented by vector dot product when valuating Im<S2|T>
    S1 = (k0/1j/Z) * np.concatenate((orient_Exfield[design_mask], orient_Eyfield[design_mask]))
    S2 = np.conj(S1) #S2 = G* @ J = S1*

    #######################set up optimization##################
    O_lin = (Z/2/k0) * (1j/2) * (Gddinv.conj().T @ S2) * dx * dy
    O_quad = sp.csc_matrix(Gddinv.shape, dtype=np.complex)

    include = np.ones(2*len(UPlist), dtype=np.bool)

    print('starting dual optimization', flush=True)
    
    optLags, optgrad, dualval, objval = get_Msparse_bound(S1, O_lin, O_quad, GinvdagPdaglist, UPlist, include, dualconst=vacPrad, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period, alg=alg)
    
    LDOS_enh = dualval / vacPrad
    print('vacPrad', vacPrad)
    print('LDOS_enh', LDOS_enh)
    
    return LDOS_enh

chi=2+3j
wvlgth=1
orient=np.array([1,1])
design_x=np.double(40)
design_y=np.double(40)
dist =1
pml_sep=np.double(1)
pml_thick=4
gprx=2
gpry=2
NProjx=1
NProjy=1
get_TE_dipole_oneside_LDOS_Msparse_bound(chi, wvlgth, orient, design_x, design_y, dist, pml_sep, pml_thick, gprx, gpry, NProjx, NProjy)