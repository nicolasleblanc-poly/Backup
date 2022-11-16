import numpy as np
import scipy.linalg as la
from .spatialProjopt_partialDual_singlematrix import get_inc_spatialProjopt_partialDual_singlematrix, get_spatialProjopt_partialDual_singlematrix
from .spatialProjopt_multiSource_Zops_singlematrix_numpy import Z_TT, grad_Z_TT



def get_spatialProjopt_generalP_partialDual(n_S, PLags, PLagsgrad, PLagsHess, Pdof, Pdofgrad, O_quad_st, S_st, U, Ideslist, P_ZTS_Sfunc, P_gradZTS_Sfunc, include=None, dualconst=0.0, zeta_xtol=1e-6, zeta_rtol=1e-4):
    """
    evaluates the dual value given a generalP constraint constructed via Pdof and Ideslist, with Lagrangian multipliers in PLags
    PLags[0] is multiplier for global SymU constraint, PLags[1] is multiplier for SymUP constraint, PLags[2] is multiplier for AsymUP constraint
    the AsymU constraint is taken care of by the partial dual
    """
    n_basis = U.shape[0]
    n_cplx_Lags = n_S**2
    
    AsymU = (U - U.conj().T)/(2j)
    eigw, eigv = la.eigh(AsymU)
    invsqrtAsymU = eigv @ np.diag(1.0/np.sqrt(eigw)) @ eigv.conj().T
    E = np.kron(np.eye(n_S), AsymU)
    invsqrtE = np.kron(np.eye(n_S), invsqrtAsymU)

    
    get_T = Pdofgrad.size>0 #need the Tvector to evaluate gradient for general P entries

    generalP = np.zeros_like(U, dtype=np.complex)
    #print('len(Ideslist)', len(Ideslist))
    #print('len(Pdof)', len(Pdof))
    
    for i in range(len(Pdof)//2):
        #print('i', i)
        generalP += (Pdof[2*i] + 1j*Pdof[2*i+1]) * Ideslist[i]

    Pdeslist = [generalP]
    UPlist = [U @ generalP]
    
    if include is None:
        include = [True, True] * n_cplx_Lags
    
    ZTTfunc = lambda L: Z_TT(n_S, L, O_quad_st, UPlist)
    gradZTTfunc = lambda L: grad_Z_TT(n_S, L, UPlist)
    
    ZTS_Sfunc = lambda L, SI: P_ZTS_Sfunc(L, SI, Pdeslist)
    gradZTS_Sfunc = lambda L, SI: P_gradZTS_Sfunc(L, SI, Pdeslist)

    if get_T:
        dual, flagRegular, flagzeta0, Tvec = get_spatialProjopt_partialDual_singlematrix(PLags, PLagsgrad, PLagsHess, S_st, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include, get_T=True, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)
    else:
        dual, flagRegular, flagzeta0 = get_spatialProjopt_partialDual_singlematrix(PLags, PLagsgrad, PLagsHess, S_st, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include, get_T=False, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    if Pdofgrad.size>0:
        
        Pdofgrad[:] = 0
        for i in range(n_S):
            for j in range(n_S):
                gamma = PLags[i*n_S + j]
                zeta = PLags[n_cplx_Lags + i*n_S + j]
                Si = S_st[i*n_basis:(i+1)*n_basis]
                Ti = Tvec[i*n_basis:(i+1)*n_basis]
                Tj = Tvec[j*n_basis:(j+1)*n_basis]
                
                for l in range(len(Ideslist)):
                    cstrt = np.vdot(Si, Ideslist[l] @ Tj) - np.vdot(Ti, U @ (Ideslist[l] @ Tj))
                    re_cstrt = np.real(cstrt)
                    im_cstrt = np.imag(cstrt)
                    Pdofgrad[2*l] += gamma*re_cstrt + zeta*im_cstrt
                    Pdofgrad[2*l+1] += zeta*re_cstrt - gamma*im_cstrt

    print('calculated partial dual is', dual)
    if Pdofgrad.size>0:
        print('norm of Pdofgrad is', la.norm(Pdofgrad))
    return dual



def get_spatialProjopt_generalP_withGlobalU_partialDual(n_S, PLags, PLagsgrad, PLagsHess, Pdof, Pdofgrad, O, S, U, Ideslist, P_ZTS_Sfunc, P_gradZTS_Sfunc, dualconst=0.0, zeta_xtol=1e-6, zeta_rtol=1e-4):
    #######CURRENTLY THIS METHOD ONLY SUPPORTS n_S=1 and empirically is unfavored as compared with get_spatialProjopt_generalP_partialDual
    """
    evaluates the dual value given a generalP constraint constructed via Pdof and Ideslist, with Lagrangian multipliers in PLags
    PLags[0] is multiplier for global SymU constraint, PLags[1] is multiplier for SymUP constraint, PLags[2] is multiplier for AsymUP constraint
    the AsymU constraint is taken care of by the partial dual
    """
    AsymU = (U - U.conj().T)/(2j)
    eigw, eigv = la.eigh(AsymU)
    invsqrtAsymU = eigv @ np.diag(1.0/np.sqrt(eigw)) @ eigv.conj().T

    get_T = Pdofgrad.size>0 #need the Tvector to evaluate gradient for general P entries

    generalP = np.zeros_like(U, dtype=np.complex)
    for i in range(len(Pdof)//2):
        generalP += (Pdof[2*i] + 1j*Pdof[2*i+1]) * Ideslist[i]

    Pdeslist = [np.eye(U.shape[0]), generalP]
    UPlist = [U, U @ generalP]
    include = [True, False, True, True] #partial dual covers the Asym(U) constraint

    ZTTfunc = lambda L: Z_TT(n_S, L, O, UPlist)
    gradZTTfunc = lambda L: grad_Z_TT(n_S, L, UPlist)
    
    ZTS_Sfunc = lambda L, SI: P_ZTS_Sfunc(L, SI, Pdeslist)
    gradZTS_Sfunc = lambda L, SI: P_gradZTS_Sfunc(L, SI, Pdeslist)

    if get_T:
        dual, flagRegular, flagzeta0, Tvec = get_inc_spatialProjopt_partialDual_singlematrix(include, PLags, PLagsgrad, PLagsHess, S, AsymU, invsqrtAsymU, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, get_T=True, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)
    else:
        dual, flagRegular, flagzeta0 = get_inc_spatialProjopt_partialDual_singlematrix(include, PLags, PLagsgrad, PLagsHess, S, AsymU, invsqrtAsymU, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, get_T=False, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    if Pdofgrad.size>0:
        for i in range(len(Ideslist)):
            cstrt = np.vdot(S, Ideslist[i] @ Tvec) - np.vdot(Tvec, U @ (Ideslist[i] @ Tvec))
            re_cstrt = np.real(cstrt)
            im_cstrt = np.imag(cstrt)
            Pdofgrad[2*i] = PLags[1]*re_cstrt + PLags[2]*im_cstrt
            Pdofgrad[2*i+1] = PLags[2]*re_cstrt - PLags[1]*im_cstrt

    print('calculated partial dual is', dual)
    if Pdofgrad.size>0:
        print('norm of Pdofgrad is', la.norm(Pdofgrad))
    return dual

