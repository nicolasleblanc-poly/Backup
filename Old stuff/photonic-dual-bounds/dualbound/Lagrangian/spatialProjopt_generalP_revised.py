import numpy as np
import scipy.linalg as la
from .spatialProjopt_partialDual_revised import get_inc_spatialProjopt_partialDual_revised, get_spatialProjopt_partialDual_revised
from .spatialProjopt_multiSource_Zops_singlematrix_numpy import Z_TT, grad_Z_TT



def get_spatialProjopt_generalP_partialDual_revised(n_S, PLags, PLagsgrad, PLagsHess, Pdof, Pdofgrad, O_quad_st, S_st, U, Ideslist, UP_ZTTfunc, UP_gradZTTfunc, P_ZTS_Sfunc, P_gradZTS_Sfunc, include=None, dualconst=0.0, zeta_xtol=1e-10, zeta_rtol=1e-10, ktol=1e-4):
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
    
    #ZTTfunc = lambda L: Z_TT(n_S, L, O_quad_st, UPlist)
    #gradZTTfunc = lambda L: grad_Z_TT(n_S, L, UPlist)

    ZTTfunc = lambda L: UP_ZTTfunc(L, UPlist)
    gradZTTfunc = lambda L: UP_gradZTTfunc(L, UPlist)
    
    ZTS_Sfunc = lambda L, SI: P_ZTS_Sfunc(L, SI, Pdeslist)
    gradZTS_Sfunc = lambda L, SI: P_gradZTS_Sfunc(L, SI, Pdeslist)

    dual, pDData = get_spatialProjopt_partialDual_revised(PLags, PLagsgrad, PLagsHess, S_st, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol, ktol=ktol)

    
    if Pdofgrad.size>0:

        T = pDData['T']
        flagRegular = pDData['flagRegular']
        if not flagRegular:
            invsqrtEunitk = pDData['invsqrtEunitk']
            globalCvxCstrt = pDData['globalCvxCstrt']

        UIlist = []
        for l in range(len(Ideslist)):
            UIlist.append(U @ Ideslist[l])
            
        Pdofgrad[:] = 0
        for i in range(n_S):
            for j in range(n_S):
                gamma = PLags[i*n_S + j]
                zeta = PLags[n_cplx_Lags + i*n_S + j]
                Si = S_st[i*n_basis:(i+1)*n_basis]
                Ti = Tvec[i*n_basis:(i+1)*n_basis]
                Tj = Tvec[j*n_basis:(j+1)*n_basis]
                if not flagRegular:
                    invsqrtEunitk_i = invsqrtEunitk[i*n_basis:(i+1)*n_basis]
                    invsqrtEunitk_j = invsqrtEunitk[j*n_basis:(j+1)*n_basis]
                
                for l in range(len(Ideslist)):
                    UI = UIlist[l]
                    cstrt = np.vdot(Si, Ideslist[l] @ Tj) - np.vdot(Ti, UI @ Tj)
                    reLag_grad = np.real(cstrt)
                    imLag_grad = np.imag(cstrt)
                    if not flagRegular: #kernel solution, need extra set constraint contribution
                        reLag_zetaKgrad = -np.real(np.vdot(invsqrtEunitk_i, UI @ invsqrtEunitk_j))
                        reLag_grad += reLag_zetaKgrad * globalCvxCstrt
                        imLag_zetaKgrad = -np.imag(np.vdot(invsqrtEunitk_i, UI @ invsqrtEunitk_j))
                        imLag_grad += imLag_zetaKgrad * globalCvxCstrt
                        
                    Pdofgrad[2*l] += gamma*reLag_grad + zeta*imLag_grad
                    Pdofgrad[2*l+1] += zeta*reLag_grad - gamma*imLag_grad

    print('calculated partial dual is', dual)
    if Pdofgrad.size>0:
        print('norm of Pdofgrad is', la.norm(Pdofgrad))
    return dual


