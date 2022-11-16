import numpy as np
import scipy.sparse.linalg as spla
import scipy.sparse as sp
def get_pml_x(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20):
    pml_x_Hz = np.ones((Nx,Ny), dtype=np.complex128)
    # print("pml_x_Hz",pml_x_Hz,"\n")
    # print("pml_x_Hz[:Npmlx,:]",pml_x_Hz[:Npmlx,:],"\n")
    pml_x_Ey = np.ones((Nx,Ny), dtype=np.complex128)
    if Npmlx==0:
        return pml_x_Hz, pml_x_Ey
    x = np.arange(Nx)
    y = np.arange(Ny)
    X,Y = np.meshgrid(x,y, indexing='ij')
    w_pml = Npmlx * dx
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    #define left PML
    x = X[:Npmlx, :]
    pml_x_Hz[:Npmlx,:] = 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-x) / Npmlx)**m)
    # print("pml_x_Hz[:Npmlx,:]",pml_x_Hz,"\n")
    pml_x_Ey[:Npmlx,:] = 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-x+0.5) / Npmlx)**m)
    # print("pml_x_Hz[1:Npmlx,:] 4",pml_x_Ey[1:Npmlx,:],"\n")
    #define right PML
    x = X[-Npmlx:, :]
    # print("x",x,"\n")
    pml_x_Hz[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx)/Npmlx)**m)
    # print("pml_x_Hz[1:Npmlx,:] 5",pml_x_Hz[-Npmlx:,:],"\n")
    pml_x_Ey[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx-0.5)/Npmlx)**m)
    # print("pml_x_Hz[1:Npmlx,:] 6",pml_x_Ey[-Npmlx:,:],"\n")
    return pml_x_Hz, pml_x_Ey
def get_pml_y(omega, Nx, Ny, Npmly, dy, m=3, lnR=-20):
    pml_y_Hz = np.ones((Nx,Ny), dtype=np.complex128)
    pml_y_Ex = np.ones((Nx,Ny), dtype=np.complex128)
    if Npmly==0:
        return pml_y_Hz, pml_y_Ex
    x = np.arange(Nx)
    y = np.arange(Ny)
    X,Y = np.meshgrid(x,y, indexing='ij')
    w_pml = Npmly * dy
    sigma_max = -(m+1)*lnR / (2*w_pml) / omega
    #define bottom PML
    y = Y[:,:Npmly]
    # print("y",y-2)
    pml_y_Hz[:,:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((Npmly-y)/Npmly)**m)
    pml_y_Ex[:,:Npmly] = 1.0 / (1.0 + 1j * sigma_max * ((Npmly-y-0.5)/Npmly)**m)
    #define top PML
    y = Y[:,-Npmly:]
    pml_y_Hz[:,-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y-Ny+1+Npmly)/Npmly)**m)
    pml_y_Ex[:,-Npmly:] = 1.0 / (1.0 + 1j * sigma_max * ((y-Ny+1+Npmly+0.5)/Npmly)**m)
    return pml_y_Hz, pml_y_Ex
def build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy):
    """
    construct TE FDFD system matrix for vacuum
    the ordering of the indices goes (x,y,Hz), (x,y,Ex), (x,y,Ey), (x,y+1,Hz), (x,y+1,Ex) , ...
    for now default periodic boundary conditions with no phase shift
    """
    pml_x_Hz, pml_x_Ey = get_pml_x(omega, Nx, Ny, Npmlx, dx)
    pml_y_Hz, pml_y_Ex = get_pml_y(omega, Nx, Ny, Npmly, dy)
    A_data = []
    A_i = []
    A_j = []  #prepare to construct A matrix in COO format
    for cx in range(Nx):
        for cy in range(Ny):
            xyind = cx*Ny + cy
            if cx<Nx-1:
                xp1yind = (cx+1)*Ny + cy
            else:
                xp1yind = cy
            if cx>0:
                xm1yind = (cx-1)*Ny + cy
            else:
                xm1yind = (Nx-1)*Ny + cy
            if cy<Ny-1:
                xyp1ind = cx*Ny + cy + 1
            else:
                xyp1ind = cx*Ny
            if cy>0:
                xym1ind = cx*Ny + cy - 1
            else:
                xym1ind = cx*Ny + Ny - 1
            Hzind = 3*xyind
            #construct Hz row
            i = Hzind
            A_i.append(i); A_j.append(i); A_data.append(-1j*omega) #diagonal
            jEx0 = 3*xym1ind + 1
            jEx1 = i + 1
            A_i.append(i); A_j.append(jEx0); A_data.append(pml_y_Hz[cx,cy]/dy)
            A_i.append(i); A_j.append(jEx1); A_data.append(-pml_y_Hz[cx,cy]/dy) #Ex part of curl E term
            jEy0 = i + 2
            jEy1 = 3*xp1yind + 2
            A_i.append(i); A_j.append(jEy0); A_data.append(-pml_x_Hz[cx,cy]/dx) #Ey part of curl E term
            A_i.append(i); A_j.append(jEy1); A_data.append(pml_x_Hz[cx,cy]/dx)
            #construct Ex row
            i = i+1 #Ex comes after Hz
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            jHz0 = Hzind
            jHz1 = 3*xyp1ind
            A_i.append(i); A_j.append(jHz0); A_data.append(-pml_y_Ex[cx,cy]/dy)
            A_i.append(i); A_j.append(jHz1); A_data.append(pml_y_Ex[cx,cy]/dy) #Hz curl
            #constraint Ey row
            i = i+1 #Ey comes after Ex
            A_i.append(i); A_j.append(i); A_data.append(1j*omega)
            jHz0 = 3*xm1yind
            jHz1 = Hzind
            A_i.append(i); A_j.append(jHz0); A_data.append(pml_x_Ey[cx,cy]/dx)
            A_i.append(i); A_j.append(jHz1); A_data.append(-pml_x_Ey[cx,cy]/dx) #Hz curl
    A = sp.coo_matrix((A_data, (A_i,A_j)), shape=(3*Nx*Ny,3*Nx*Ny))
    return A.tocsc()
def get_diagA_from_chigrid(omega,chi_x,chi_y):
    Nx,Ny=chi_x.shape
    diagA=np.zeros(3*Nx*Ny,dtype=np.complex128)
    diagA[1::3]=1j*omega*chi_x.flatten()
    diagA[2::3]=1j*omega*chi_y.flatten()
    return sp.diags(diagA)
# Code to generate the fields
def get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx,cy, pol, amp=1.0, Qabs=np.inf, chigrid=None):
    omega = 2*np.pi/wvlgth * (1 + 1j/2/Qabs)
    A = build_TE_vac_A(omega, Nx, Ny, Npmlx, Npmly, dx, dy)
    # print("A",A,"\n")
    if not (chigrid is None):
        A += get_diagA_from_chigrid(omega, chigrid, chigrid)
    b = np.zeros(A.shape[0], dtype=np.complex128)
    xyind = cx*Ny + cy
    b[3*xyind+pol] = amp
    # print("b",b,"\n")
    x = spla.spsolve(A,b)
    # print("x",x,"\n")
    Hzfield = np.reshape(x[::3], (Nx,Ny))
    Exfield = np.reshape(x[1::3], (Nx,Ny))
    Eyfield = np.reshape(x[2::3], (Nx,Ny))
    return Hzfield, Exfield, Eyfield
# Code to get the Green's function associated to the fields 
def get_Yee_TE_GreenFcn(wvlgth, Gx,Gy, Npmlx, Npmly, dx,dy, Qabs=np.inf):
    """
    generate Green's function of a domain with shape (Gx,Gy), with a periodicY unit cell of Y height Ny
    Green's function basis ordering: ...(x,y,Ex)...(0,0,Ey),(0,1,Ey),...,(x,y,Ey),...
    """
    gpwx = int(1.0/dx)
    gpwy = int(1.0/dy)
    Nx = 2*Gx-1 + gpwx//2 + 2*Npmlx
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly

    cx = Nx//2
    # print("Cx",cx,"\n")
    cy = Ny//2
    # print("Cy",cy,"\n")
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, Qabs=Qabs)
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, Qabs=Qabs)
    # print("x_Exfield ",x_Exfield,"\n")
    # print("x_Eyfield ",x_Eyfield,"\n")
    # print("y_Exfield ",y_Exfield,"\n")
    # print("y_Eyfield ",y_Eyfield,"\n")
    # The 1 and 2 is the value of the variable pol 
    numCoord = Gx*Gy
    G = np.zeros((2*numCoord,2*numCoord), dtype=np.complex128)
    for ix in range(Gx):
        for iy in range(Gy):
            x_Ex = x_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            x_Ey = x_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ex = y_Exfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            y_Ey = y_Eyfield[cx-ix:cx-ix+Gx,cy-iy:cy-iy+Gy]
            xyind = ix*Gy + iy
            # print("ix",ix,"\n")
            # print("iy",iy,"\n")
            # print("xyind",xyind,"\n")
            # print("x_Ex.flatten()",x_Ex.flatten(),"\n")
            # 4x1 vector
            # print("G[1:numCoord-1,xyind]",G[:numCoord,xyind],"\n")
            # print("numCoord ",numCoord,"\n")
            # print("xyind ",xyind,"\n")
            G[:numCoord,xyind] = x_Ex.flatten()
            # print("G ",G,"\n")
            G[numCoord:,xyind] = x_Ey.flatten()
            G[:numCoord,xyind+numCoord] = y_Ex.flatten()
            G[numCoord:,xyind+numCoord] = y_Ey.flatten()
    print("x_Ex.flatten()", x_Ex.flatten(), "\n")
    orient_Exfield =  x_Ex.flatten() +  y_Ex.flatten()
    orient_Eyfield =  x_Ey.flatten() +  y_Ey.flatten()
    print("orient_Exfield.shape", orient_Exfield.shape, "\n")
    # print("orient_Eyfield.shape", orient_Eyfield.shape, "\n")
    
    ei = np.zeros((2*len(orient_Exfield),1))
    ei[:len(orient_Exfield),0] = orient_Exfield
    ei[len(orient_Exfield):,0] = orient_Eyfield # ei is a 12x1 vector here
    print("ei 1", ei, "\n")
    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    return G, ei
