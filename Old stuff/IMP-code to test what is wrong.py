# Import necessary librairies
import math 
import numpy as np
import scipy
import scipy.sparse.linalg as spla
import scipy.sparse as sp
from timeit import default_timer as timer
import copy
import scipy.linalg as la
import matplotlib.pyplot as plt
import subprocess
import os
def get_pml_x(omega, Nx, Ny, Npmlx, dx, m=3, lnR=-20):
    pml_x_Hz = np.ones((Nx,Ny), dtype=np.complex128)
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
    pml_x_Ey[:Npmlx,:] = 1.0 / (1.0 + 1j * sigma_max * ((Npmlx-x+0.5) / Npmlx)**m)
    #define right PML
    x = X[-Npmlx:, :]
    # print("x",x,"\n")
    pml_x_Hz[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx)/Npmlx)**m)
    pml_x_Ey[-Npmlx:,:] = 1.0 / (1.0 + 1j * sigma_max * ((x-Nx+1+Npmlx-0.5)/Npmlx)**m)
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
    if not (chigrid is None):
        A += get_diagA_from_chigrid(omega, chigrid, chigrid)
    b = np.zeros(A.shape[0], dtype=np.complex128)
    xyind = cx*Ny + cy
    b[3*xyind+pol] = amp
    x = spla.spsolve(A,b)
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
    print("Nx",Nx,"\n")
    Ny = 2*Gy-1 + gpwy//2 + 2*Npmly
    print("Ny",Ny,"\n")
    cx = Nx//2
    print("cx",cx,"\n")
    cy = Ny//2
    print("cy",cy,"\n")
    _, x_Exfield, x_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 1, Qabs=Qabs)
    _, y_Exfield, y_Eyfield = get_TE_dipole_field(wvlgth, Nx, Ny, Npmlx, Npmly, dx, dy, cx, cy, 2, Qabs=Qabs)

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
            G[:numCoord,xyind] = x_Ex.flatten()
            G[numCoord:,xyind] = x_Ey.flatten()
            G[:numCoord,xyind+numCoord] = y_Ex.flatten()
            G[numCoord:,xyind+numCoord] = y_Ey.flatten()
    ei = np.zeros((2*numCoord,1), dtype=np.complex128)
    print("ei.shape", ei.shape,"\n")
    orient_Exfield = 1 * x_Ex.flatten()+ 1* y_Ex.flatten()
    orient_Eyfield = 1* x_Ey.flatten() + 1 * y_Ey.flatten()
    print("len(x_Ex)", len(x_Ex),"\n")
    print("orient_Exfield", orient_Exfield.shape, "\n")
    for i in range(len(orient_Exfield)):
        ei[i] = orient_Exfield[i]
    for j in range(len(orient_Exfield), len(orient_Eyfield)+len(orient_Exfield)):
        ei[i] = orient_Eyfield[i]

    # ei = -0.5 * np.real(1 * orient_Exfield + 1 * orient_Eyfield)
    print("ei.shape", ei.shape,"\n")
    print("G.shape", G.shape,"\n")

    eta = 1.0 #dimensionless units
    k0 = 2*np.pi/wvlgth * (1+1j/2/Qabs) / 1
    Gfac = -1j*k0/eta
    G *= Gfac
    return G, ei
def DipoleOverSlab_setup(dx,dy,addx,liste,dL):
    Mx = dx+addx # We will have 3 more pixels on the left side of the region
    # used to generate the Green's function. The dipole will be 
    # situated in this region of length 3 pixels.
    My = dy
    Npml = 15
    dL=0.05
    Qabs = 1e4
    # Obtain the Green's function
    fct_call = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs=Qabs)
    Gbig = fct_call[0]
    ei = fct_call[1]
    print("Gbig",Gbig,"\n")
    print("ei",ei,"\n")
    # # The shape of the Green's function is (2*M_x*M_y,2*M_x*M_y)
    # # print("The size of the Green's function is ",Gbig.shape,"\n")
    # # Code to cut the matrix from 920x920 to 800x800 but only considering 
    # # the positions 61x->61 to 460x->460 and 61y->521 to 460y->920. 
    # # The matrix M is the Green's function for the domain of interest.
    # dim=2*dx*dy # 2*dx*dy 
    # # M=np.zeros((dim,dim),dtype='complex')
    # i=addx*dy # -1 on the index, since python starts indexing at 0 and not 1.
    # # icopy=copy.deepcopy(i)
    # # print("Gbig",Gbig)
    # j=int(len(Gbig)/2)+addx*dy
    # # # print("i",i,"\n")
    # # # print("j",j,"\n")
    # # jcopy=copy.deepcopy(j)
    # position=int(len(Gbig)/2)
    # # print("position",position,"\n")
    # half_lenGbig=int(len(Gbig)/2)
    # lenGbig=int(len(Gbig))

    # dipx=int(i/2)-1
    # dipy=half_lenGbig+dipx
    # u=np.zeros((dim,1),dtype='complex')
    # v=np.zeros((dim,1),dtype='complex')
    # u[:position,0]=Gbig[i:half_lenGbig,dipx]
    # u[position:,0]=Gbig[j:lenGbig,dipx]
    # v[:position,0]=Gbig[i:half_lenGbig,dipy]
    # v[position:,0]=Gbig[j:lenGbig,dipy]
    # alpha=1/(math.sqrt(2))
    # beta=1/(math.sqrt(2))
    # # coeff=-(z*1j)/(2*k0)
    # ei=alpha*u+beta*v # this gives ei for a dipole at 45 degrees
    # # ei=coeff*(alpha*u+beta*v)
    # # print(ei.shape) # 800x1 vector
    
    # print(ei_tr.shape) # 1x800 vector
    Pv=[]
    for element in liste:
        # print("element",element)
        P=np.diag(element)
        Pv.append(P)
        # print("Pv",Pv)
    # print("Pv",Pv,"\n")
    return Gbig,ei,Pv
# First constraint
def C1(T,i,ei_tr, chi_invdag, Gdag,Pv):
    # Left term
    P=Pv[i]
    # print("P shape",P.shape,"\n")
    # print("T shape",T,"\n")
    PT=np.matmul(P,T)
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=np.matmul(ei_tr,PT)
    I_EPT = np.imag(EPT) # (1/2)*
    # Right term
    chiGP = np.matmul(P,chi_invdag - Gdag)
    chiG_A=(chiGP-np.matrix.conjugate(np.transpose(chiGP)))/(2j)
    # M^A = (M+M^dagg)/2 -> j is i in python
    chiGA_T = np.matmul(chiG_A,T)
    T_chiGA_T = np.matmul(np.matrix.conjugate(np.transpose(T)),chiGA_T)
    return I_EPT - T_chiGA_T
# Second constraint
def C2(T,i,ei_tr, chi_invdag, Gdag,Pv):
    # Left term
    P=Pv[i]
    PT=np.matmul(P,T)
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=np.matmul(ei_tr,PT)
    I_EPT = np.real(EPT) # (1/2)*
    # Right term
    chiGP = np.matmul(P,chi_invdag - Gdag)
    # M^S = (M+M^dagg)/2 
    chiG_S=(chiGP+np.matrix.conjugate(np.transpose(chiGP)))/2
    chiGS_T = np.matmul(chiG_S,T)
    T_chiGS_T = np.matmul(np.matrix.conjugate(np.transpose(T)),chiGS_T)
    return I_EPT - T_chiGS_T
# Code for the creation of the A matrix
def Am(x, chi_invdag, Gdag,Pv): # the A matrix
        A=0.0+0.0j
        for i in range(len(x)):
            P=Pv[math.floor(i/2)]
            if i % 2 ==0: # Anti-sym
                chiGP = np.matmul(P,chi_invdag - Gdag)
                chiG_A=(chiGP-np.matrix.conjugate(np.transpose(chiGP)))/(2j)
                A += x[i]*chiG_A
            else: # Sym
                chiGP = np.matmul(P,chi_invdag - Gdag)
                chiG_S=(chiGP+np.matrix.conjugate(np.transpose(chiGP)))/(2)
                A += x[i]*chiG_S
        # print(A)
        return A
# Code for the creation of the b vector used to solve for T in the Ax=b system.
def bv(x,ei,Pv):
        term1 = k0*1.0j/(8*Z)
        b = term1*ei
        for i in range(len(x)):
            P=Pv[math.floor(i/2)]
            term = (1/4)*x[i]*P
            b += np.matmul(term,ei)
#             if i % 2 ==0:
#                 term = (1/4)*x[i]*P
#                 b += np.matmul(term,ei)
#             else:
#                 term = (1/4)*x[i]*P
#                 b += np.matmul(term,ei)
        return b
# Code for the different T solvers
def th_solve(A,b):
    T=np.linalg.solve(A,b)
    return T,A
def cg_solve(A,b):
    T=spla.cg(A,b,tol=1e-5)[0]
    return T,A
def bicg_solve(A,b):
    T=spla.bicg(A,b,tol=1e-5)
    return T
def bicgstab_solve(A,b):
    T=spla.bicgstab(A,b,tol=1e-5)
    return T
def gmres_solve(A,b):
    T=spla.gmres(A,b,tol=1e-5)
    return T
tsolvers = {
    "th": th_solve,
    "cg": cg_solve,
    "bicg": bicg_solve,
    "bicgstab": bicgstab_solve,
    "gmres": gmres_solve
}
def tsolverpick(name,A,b):
    return tsolvers[name](A,b)
def Dual(x,g,fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad):
    # Ask Pengning about the question below: 
    # I never use the g input. Should I use it to construct the 
    # A matrix and the b vector? How to split the total gradient value
    # (aka the contraint value) since it has two terms.
    A=Am(x, chi_invdag, Gdag, Pv)
    b=bv(x,ei, Pv)
    solve=tsolverpick(tsolver,A,b)
    # print("x1 ",x,"\n")
    T=solve[0]
    # print("T shape",T.shape,"\n")
    # A=solve[1]
    g=np.ones(len(x),dtype='complex')
    for i in range(len(x)):
        if i % 2 == 0:
            g[i] = C1(T,math.floor(i/2),ei_tr, chi_invdag, Gdag, Pv) 
        else:
            g[i] = C2(T,math.floor(i/2),ei_tr, chi_invdag, Gdag, Pv) 
    D=0.0+0.0j
    if len(fSlist) == 0:
        ei_T=np.matmul(ei_tr,T)
        obj = ((-1/2)*np.real(ei_T*k0*1.0j/(Z)))+0.0j
        D = obj
        for i in range(len(x)):
            D+=x[i]*g[i]
    else:
        if isinstance(fSlist, list):
            f=fSlist[0]
        else: # fSlist is an array
            f=fSlist
        A_f = np.matmul(A,f)
        f_tr = np.matrix.conjugate(np.transpose(f)) 
        fAf=np.matmul(f_tr,A_f)
        ei_T=np.matmul(ei_tr,T)
        obj = ((-1/2)*np.real(ei_T*k0*1.0j/(Z)))+fAf
        D = obj
        for i in range(len(x)):
            D+=x[i]*g[i]
    # print("x2 ",x,"\n")
    # print("g",g,"\n")
    # print("obj ",obj,"\n")       
    if get_grad == True:
        return D, g, obj # D.real
    elif get_grad == False:
        return D, obj # D.real
def mineigfunc(x, chi_invdag, Gdag, Pv):
    A=Am(x, chi_invdag, Gdag, Pv)
    # print(A)
    w_val,v=la.eig(A)
    # w: eigen values
    # v: eigen vectors
    w=np.real(w_val)
    # This code is to find the minimum eigenvalue
    min_eval=w[-1]
    # print(min_eval)
    # This code is to find the eigenvector
    # associated with the mimimum eigenvalue
    dim=A.shape[0]
    min_evec=v[:,-1].reshape(dim,1)
    return min_eval,min_evec
def validityfunc(x, chi_invdag, Gdag, Pv):
    val=mineigfunc(x, chi_invdag, Gdag, Pv)
    min_eval=val[0]
    min_evec=val[1]
    if min_eval>0:
        return 1
    else:
        return -1
def BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, gradConverge=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    dofnum = len(initdof)
    grad = np.zeros(dofnum)
    tmp_grad = np.zeros(dofnum)
    Hinv = np.eye(dofnum) #approximate inverse Hessian
    prev_dualval = np.inf
    dof = initdof
    justfunc = lambda d: dgfunc(d, np.array([]), fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=False)
    olddualval = np.inf
    reductCount = 0
    it=0
    while True: #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = np.eye(dofnum,dtype='complex') #reset Hinv
        val = dgfunc(dof, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)  
        dualval = val[0]
        grad = val[1]
    #             print("grad",grad,"\n")
                # This gets the value of the dual function and the dual's gradient
    #             print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
        while True:
            # print("Hinv",Hinv,"\n")
            iternum += 1
    #                 print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)
            Ndir = - Hinv @ grad
            pdir = Ndir / la.norm(Ndir)
                    #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = pdir @ grad
    #                 print('pdir dot grad is', p_dot_grad)
            c_reduct = 0.7; c_A = 1e-4; c_W = 0.9
            alpha = alpha_start = alpha_last
    #                 print('starting alpha', alpha_start)
            while validityfunc(dof+alpha*pdir, chi_invdag, Gdag, Pv)<=0: #move back into feasibility region
                alpha *= c_reduct
            alpha_feas = alpha
    #                 print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = np.inf
            while True:
                tmp_dual = justfunc(dof+alpha*pdir)[0]
                # print("tmp_dual",tmp_dual,"\n")
                # print("Dopt",Dopt,"\n")
                # print("In here bud \n")
                if tmp_dual<Dopt: #the dual is still decreasing as we backtrack, continue
                    # print("blabla")
                    Dopt = tmp_dual; alphaopt=alpha
                else:
                    alphaopt=alpha ###ISSUE!!!!
                    break
                if tmp_dual<=dualval + c_A*alpha*p_dot_grad: #Armijo backtracking condition
                    alphaopt = alpha
                    break
                alpha *= c_reduct
            added_fakeS = False
            if alphaopt/alpha_start>(c_reduct+1)/2: #in this case can start with bigger step
                alpha_last = alphaopt*2
            else:
                alpha_last = alphaopt
                if alpha_feas/alpha_start<(c_reduct+1)/2 and alphaopt/alpha_feas>(c_reduct+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = True
    #                         print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + alpha_feas*pdir #dof that is roughly on duality boundary
                    eig_fct_eval = mineigfunc(singular_dof, chi_invdag, Gdag, Pv) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[0]
                    mineigv=eig_fct_eval[1]
    #                         print("min_evec shape",mineigv.shape,"\n")
    #                         print("min_evec type",type(mineigv),"\n")
                            # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeSval = dgfunc(dof, np.array([]), [mineigv], tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=False)[0]
                            # This only gets us the value of the dual function
                    epsS = np.sqrt(fakeSratio*np.abs(dualval/fakeSval))
    #                         print('epsS', epsS, '\n')
                            #fSlist.append(np.matmul(epsS,mineigv))
                    fSlist.append(epsS*mineigv) #add new fakeS to fSlist
    #                         print('length of fSlist', len(fSlist))
    #                 print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir
                    #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta, tmp_grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
            tmp_dual = tmp_val[0]
            tmp_grad = tmp_val[1]
            p_dot_tmp_grad = pdir @ tmp_grad
            if added_fakeS:
                Hinv = np.eye(dofnum,dtype='complex') #the objective has been modified; restart Hinv from identity
            elif p_dot_tmp_grad > c_W*p_dot_grad: #satisfy Wolfe condition, update Hinv
    #                     print('updating Hinv')
                # BFGS update
                # print("Hein?")
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                # print("gamma_dot_delta ",gamma_dot_delta ,"\n")
                Hinv_dot_gamma = Hinv@tmp_grad + Ndir
    #                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
    #                     print("Hinv",Hinv,"\n")
                Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
            dualval = tmp_dual
            grad[:] = tmp_grad[:]
            dof += delta
            objval = dualval - np.dot(dof,grad)
            eqcstval = np.abs(dof) @ np.abs(grad)
    #                 print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
    #                 print('normgrad is', la.norm(grad))
            if gradConverge and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval) and la.norm(grad)<opttol * np.abs(dualval): #objective and gradient norm convergence termination
                break
            if (not gradConverge) and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval): #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            if iternum % iter_period==0:
    #                     print('prev_dualval is', prev_dualval)
                if np.abs(prev_dualval-dualval)<np.abs(dualval)*opttol: #dual convergence / stuck optimization termination
                    break
                prev_dualval = dualval
        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break
            """
                #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
                    #break
                if len(fSlist)==0: #converged without needing fake sources, suggests strong duality
                    break
            """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    return dof, grad, dualval, objval
def item(x0, Dual, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv):
    mu=10e-8
    L=10e6
    q=mu/L
    Ak=0
    xk=x0
    yk=x0
    zk=x0
    # Where is the association with the problem at hand?
    # Where is |T> in here?
    tol = 1e-5
    cdn = False
    grad = np.zeros(len(x0))
    fSlist=[]
    indom = False
    while cdn == False and indom == False: # setup for 20 steps
        Ak=((1+q)*Ak+2*(1+math.sqrt((1+Ak)*(1+q*Ak))))/(1-q)**2
        bk=Ak/((1-q)*Ak)
        dk=((1-q**2)*Ak-(1+q)*Ak)/(2*(1+q+q*Ak))
        # store the previous xk, yk and zk values to calculate the norm
        xk_m1=xk 
        yk_m1=yk
        zk_m1=zk
        # Let's calculate the new yk 
        yk=(1-bk)*zk_m1+bk*xk_m1
        # We need to solve for |T> with the yk multipliers
        val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
        T_yk = val_yk[0] # we don't use this here but it was used to calculate the gradient below
        g_yk = val_yk[1] # this is the gradient evaluated at the |T> found with the yk multipliers
        # We can now calculate the new xk and zk values
        xk=yk-(1/L)*g_yk*yk_m1
        zk=(1-q*dk)*zk_m1+q*dk*yk_m1-(dk/L)*g_yk
        # Check if it is still necessary to go to the next iteration by
        # verifying the tolerance and if the smallest eigenvalue is positive,
        # which indicates we are in the domain. 
        if np.linalg.norm(xk-xk_m1)<tol and validityfunc(yk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True
        if np.linalg.norm(yk-yk_m1)<tol and validityfunc(xk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True
        if np.linalg.norm(zk-zk_m1)<tol and validityfunc(zk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True  
    val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_yk = val_yk[0]
    g_yk = val_yk[1]
    obj_yk = val_yk[2]
    val_xk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_xk = val_xk[0]
    g_xk = val_xk[1]
    obj_xk = val_xk[2]
    val_zk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    D_zk = val_zk[0]
    g_zk = val_zk[1]
    obj_zk = val_zk[2]
    dof = [yk, xk, zk]
    grad = [g_yk, g_xk, g_zk]
    dualval =[D_yk, D_xk, D_zk]
    objval = [obj_yk, obj_xk, obj_zk]
    print("results for yk, results for xk, results for zk")
    return dof, grad, dualval, objval
    
def Cpp_Execution(liste):
    # creating a pipe to child process
    data, temp = os.pipe()
    # writing inputs to stdin and using utf-8 to convert it to byte string
    # you can only write with strings: https://www.geeksforgeeks.org/python-os-write-method/
    os.write(temp, liste);
    os.close(temp)
    # temp is for writing and data is for reading
    # storing output as a byte string
    s = subprocess.check_output("g++ nomad.cpp -o out1;./out1", stdin = data, shell = True)
    # decoding to print a normal output
    a=s.decode("utf-8")
    # print(int(a)+2)
    print(s.decode("utf-8"))

def nomad(x0, dgfunc, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, gradConverge=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    # We need to solve for |T> with the initial multiplier values
    grad = np.zeros(len(x0))
    fSlist = [] 
    eval = Dual(x0, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    obj = eval[2][0][0]
    grad = eval[1]
    liste = [obj]
    # print("grad",grad[0])
    for i in range(len(grad)):
        liste.append(grad[i])
    # for i in grad:
    #     liste.append(grad)
    print("liste", liste, "\n")
    # The code above append to the list, which will be the input to the cpp code, in an alternating way, 
    # so it appends the value of the C1 coefficient  for some multipliers and then the value of the 
    # second constraint for the same multiplier value. This should simplify things when writing the dual 
    # function as the optimization function in the cpp code for nomad. 
    # The list has the following form:
    # liste(obj, C_1(x_0), C2_(x_0), C1(x_1), C2(x_1), ...) -> continue this for the different multiplier values x_k.

    # enter = ""

    # Driver functions
    # if __name__ == "__main__": 
    #     multipliers = Cpp_Execution(liste)

    # # Solve for the T with the new multiplier values
    # T = Dual(x0, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=False)[1]

    x=2
    return x

lsolvers = {
    "bfgs": BFGS_fakeS_with_restart, "item": item, "nomad": nomad} 
def lsolverpick(name,x,tsolver,ei, ei_tr,chi_invdag,Gdag, Pv):
    return lsolvers[name](x,Dual,validityfunc, mineigfunc,tsolver,ei, ei_tr,chi_invdag,Gdag, Pv)
# Code for the optimization of the extracted power 
def extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv):
    # Important values
    s=len(M)
    chi_coeff=3.0+0.01j
    chi=chi_coeff*np.identity(s)
    chi_invdag = np.linalg.inv(np.matrix.conjugate(np.transpose(chi)))
    ei_tr = np.matrix.conjugate(np.transpose(ei)) 
    Gdag=np.matrix.conjugate(np.transpose(M))
    print("Solve using the ",tsolver," method and the ",lsolver," method")
    start = timer()
    x_th=lsolverpick(lsolver,x0,tsolver,ei,ei_tr,chi_invdag,Gdag, Pv)
    print("Done solving")
    gradient=x_th[-3]
    dual=x_th[-2]
    objective=x_th[-1]
    mult=x_th[0]
    stop = timer()
    runtime=stop-start
    # print("x_cg",x_cg,"\n")
    print("Gradient value: ",gradient,"\n")
    print("Dual value: ",dual,"\n")
    print("Objective value: ",objective,"\n")
    print("The multipliers are: ", mult,"\n" )
    # print("Sln",x_cg,"\n")
    print("Run time: ",runtime,"seconds \n")
    return runtime
# Here are some global values
Z=376.730313668 # Impedence of free space
wvlgth = 1.0
Qabs = 1e4
omega = (2*np.pi/wvlgth) * (1+1j/2/Qabs)
k0 = omega / 1
dL = 0.05
# 2 projection matrices, so 4 constraints
# cg and bfgs
dx=2 # -> Mx
dy=2 # -> My 
addx=0
liste=[[1,1,1,1,1,1,1,1]]
# liste=[[4,4,4,4,4,4,4,4]]
# liste=[[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]]
# liste=[[1,1,1,1,1,1,1,1],[5,6,7,8,1,2,3,4]]
# liste=[[1,2,3,4,5,6,7,8],[5,6,7,8,1,2,3,4]]
setup=DipoleOverSlab_setup(dx,dy,addx,liste,dL)
print("setup",setup)
M=setup[0]
# print("M shape",M.shape,"\n")
ei=setup[1]
Pv=setup[2]
x0=np.ones(2*len(liste),dtype='complex')
tsolver="th"
lsolver="bfgs"
bfgs_time=[]
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
bfgs_time.append(opt)
tsolver="cg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
bfgs_time.append(opt)
tsolver="bicg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
bfgs_time.append(opt)
tsolver="bicgstab"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
bfgs_time.append(opt)
tsolver="gmres"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
bfgs_time.append(opt)

methods=['th', 'cg', ' bicg', 'bicgstab', ' gmres']
plt.plot(methods, bfgs_time)
plt.title("Comparison of the different T solvers when using bfgs as the multiplier solver")
plt.xlabel("The different T solvers used")
plt.ylabel("Duration of run (in seconds)")
plt.show()

tsolver="th"
lsolver="item"
item_time=[]
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
item_time.append(opt)
tsolver="cg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
item_time.append(opt)
tsolver="bicg"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
item_time.append(opt)
tsolver="bicgstab"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
item_time.append(opt)
tsolver="gmres"
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv)
item_time.append(opt)

methods=['th', 'cg', ' bicg', 'bicgstab', ' gmres']
plt.plot(methods, item_time)
plt.title("Comparison of the different T solvers when using item as the multiplier solver")
plt.xlabel("The different T solvers used")
plt.ylabel("Duration of run (in seconds)")
plt.show()
