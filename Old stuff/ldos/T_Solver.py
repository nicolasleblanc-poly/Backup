import numpy as np
import math
import scipy.sparse.linalg as spla
from A_matrix import Am
from b_vector import bv
from constraints import C1, C2
# Code for the different T solvers
def th_solve(A,b):
    T=np.linalg.solve(A,b)
    return T
def cg_solve(A,b):
    T=spla.cg(A,b,tol=1e-5)[0]
    return T
def bicg_solve(A,b):
    T=spla.bicg(A,b,tol=1e-5)[0]
    return T
def bicgstab_solve(A,b):
    T=spla.bicgstab(A,b,tol=1e-5)[0]
    return T
def gmres_solve(A,b):
    T=spla.gmres(A,b,tol=1e-5)[0]
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
def Dual(x,g,fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, get_grad, e_vac):
    # Ask Pengning about the question below: 
    # I never use the g input. Should I use it to construct the 
    # A matrix and the b vector? How to split the total gradient value
    # (aka the contraint value) since it has two terms.
    A=Am(x, chi_invdag, Gdag, Pv)
    b=bv(x,ei, Pv, e_vac)

    solve=tsolverpick(tsolver,A,b)
    # print("x1 ",x,"\n")
    T=solve
    # print("T ",T,"\n")
    # A=solve[1]
    g=np.ones(len(x),dtype='complex')
    for i in range(len(x)):
        if i % 2 == 0:
            g[i] = C1(T,math.floor(i/2),ei, chi_invdag, Gdag, Pv) 
            # g[i] = C1(T,math.floor(i/2),ei_tr, chi_invdag, Gdag, Pv) 
        else:
            g[i] = C2(T,math.floor(i/2),ei, chi_invdag, Gdag, Pv)
            # g[i] = C2(T,math.floor(i/2),ei_tr, chi_invdag, Gdag, Pv) 
    D=0.0+0.0j
    if len(fSlist) == 0:
        ei_T=np.dot(ei_tr,T)
        obj = 0.5*(0.016678204750777633+8.339102375388818e-07j)*np.imag(ei_T) + e_vac  # + vaccuum contribution
        # ((k0/(2*Z))* <- coefficient that multiplies np.imag(ei_T if needed)
        D = obj
        for i in range(len(x)):
            D += np.real(x[i]*g[i])
    else:
        if isinstance(fSlist, list):
            f=fSlist[0]
        else: # fSlist is an array
            f=fSlist
        A_f = np.matmul(A,f)
        f_tr = np.matrix.conjugate(np.transpose(f)) 
        fAf=np.matmul(f_tr,A_f)
        ei_T=np.dot(ei_tr,T)
        obj = 0.5*(0.016678204750777633+8.339102375388818e-07j)*np.imag(ei_T) + e_vac + fAf # + vacuum contribution
        # ((k0/(2*Z))*
        D = obj
        for i in range(len(x)):
            D += np.real(x[i]*g[i])
    # print("x2 ",x,"\n")
    # print("g",g,"\n")
    # print("obj ",obj,"\n")   
    # print("A.shape 1",A.shape,"\n")
    # print("b.shape 1",b.shape,"\n")    
    if get_grad == True:
        return D, g, obj, T, A, b # D.real
    elif get_grad == False:
        return D, obj, T, A, b # D.real