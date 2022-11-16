#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  8 14:48:57 2021

@author: pengning
"""

import numpy as np
import scipy.linalg as la
from scipy.optimize import root_scalar


def partialDual(T, ZTSP_S, ZTTP):
    return np.real(np.vdot(T, -ZTTP @ T)) + 2*np.real(np.vdot(T, ZTSP_S))

def get_globalcvx_constraint(S, T, E):
    return np.imag(np.vdot(S,T)) - np.real(np.vdot(T, E @ T))

def get_zetaK(ZTT, invsqrtE):
    w = la.eigvalsh(invsqrtE @ ZTT @ invsqrtE)
    return -w[0]

def get_K_mul(sigma, S, T0, E, EKinv_ZTSP_S, EKinv_S, EKinv_ET0):
    K = EKinv_ZTSP_S/sigma + 0.5j*EKinv_S - EKinv_ET0
    T = T0 + K
    return get_globalcvx_constraint(S, T, E)

def evaluate_nondefiniteZTT_kernel_solution(ZTT, ZTS_S, E, invsqrtE, S, zeta_xtol=1e-6, zeta_rtol=1e-4, kernel1D=False):
    """
    checks to see if ZTT is non-definite
    if so return zetaK the value of zeta for the kernel solution, the optimal T vector, and ZTTP
    """
    zetaK = get_zetaK(ZTT, invsqrtE)
    if zetaK<0:
        print('ZTT is definite', flush=True)
        return zetaK, -1, None
    
    eps0 = 1e-12 #tolerance for some small quantity being considered effectively 0
    ZTTP = ZTT + zetaK * E
    ZTSP_S = ZTS_S + 0.5j*zetaK*S
    
    eigw_ZTTP, eigv_ZTTP = np.linalg.eigh(ZTTP) #calculate eigenvector expansion of ZTTP for evaluating the kernel
    if eigw_ZTTP[0]<-eps0:
        print('encountered non-definite ZTTP with mineig', eigw_ZTTP[0], flush=True)
        #raise ValueError
    
    if eigw_ZTTP[0]>=eps0 or kernel1D:
        if eigw_ZTTP[0]>=eps0:
            print('zetaK solve leaves ZTTP PD beyond eps0 threshold', flush=True)
        img_mask = np.ones_like(eigw_ZTTP, dtype=np.bool)
        img_mask[0] = False
        ker_mask = np.zeros_like(eigw_ZTTP, dtype=np.bool)
        ker_mask[0] = True
    else:
        img_mask = eigw_ZTTP>=eps0
        ker_mask = eigw_ZTTP<eps0

    print('dimension of kernel space', np.sum(ker_mask))
    Ivecs = eigv_ZTTP[:, img_mask]
    Kvecs = eigv_ZTTP[:, ker_mask]
    
    ZTSP_S_img = Ivecs.conj().T @ ZTSP_S #coeffs of ZTSP_S in ZTTP eigenvector basis
    T0_img = ZTSP_S_img / eigw_ZTTP[img_mask] #ZTTP is diagonal under its eigenvector basis
    T0 = Ivecs @ T0_img
    
    E_ker = Kvecs.conj().T @ E @ Kvecs
    ZTSP_Sker = Kvecs.conj().T @ ZTSP_S
    S_ker = Kvecs.conj().T @ S
    ET0_ker = Kvecs.conj().T @ (E @ T0)
    
    EKinv_ZTSP_S = Kvecs @ la.solve(E_ker, ZTSP_Sker)
    EKinv_S = Kvecs @ la.solve(E_ker, S_ker)
    EKinv_ET0 = Kvecs @ la.solve(E_ker, ET0_ker)

    print('norm of EKinv_ZTSP_S', la.norm(EKinv_ZTSP_S))
    if la.norm(EKinv_ZTSP_S)<eps0:
        print('no coupling of ZTSP_S into kernel of ZTTP beyond eps0')
        K = np.zeros_like(T0, dtype=np.complex)
        return zetaK, partialDual(T0, ZTSP_S, ZTTP), K, T0, ZTTP
    
    #now solve for sigma multiplier to determine |K>
    zetaKcstrt_func = lambda sig: get_K_mul(sig, S, T0, E, EKinv_ZTSP_S, EKinv_S, EKinv_ET0)
    
    #first check there is a solution
    if zetaKcstrt_func(np.inf)<=0:
        print('no zetaK solution', flush=True)
        return zetaK, -1, None, None, None
    
    sigmamin = sigmamax = 1.0
    while zetaKcstrt_func(sigmamin)>0:
        sigmamin /= 2.0
    while zetaKcstrt_func(sigmamax)<=0:
        sigmamax *= 2.0
    
    sigma_sol = root_scalar(zetaKcstrt_func, bracket=[sigmamin, sigmamax], xtol=zeta_xtol, rtol=zeta_rtol, method='brentq')
    sigma = sigma_sol.root
    
    K = EKinv_ZTSP_S/sigma + 0.5j*EKinv_S - EKinv_ET0
    T = T0 + K

    zetaKdict = {} #store and return all necessary vectors for calculation of kernel solution gradient
    zetaKdict['eigw_ZTTP'] = eigw_ZTTP
    zetaKdict['eigv_ZTTP'] = eigv_ZTTP
    zetaKdict['img_mask'] = img_mask
    zetaKdict['ker_mask'] = ker_mask
    zetaKdict['Ivecs'] = Ivecs
    zetaKdict['Kvecs'] = Kvecs
    zetaKdict['EKinv'] = Kvecs @ la.inv(E_ker) @ Kvecs.conj().T
    zetaKdict['T0'] = T0
    zetaKdict['K'] = K
    zetaKdict['ZTSP_S'] = ZTSP_S
    zetaKdict['ZTTP'] = ZTTP
    zetaKdict['EKinv_ZTSP_S'] = EKinv_ZTSP_S
    zetaKdict['EKinv_S'] = EKinv_S
    zetaKdict['EKinv_ET0'] = EKinv_ET0
    zetaKdict['img_T0'] = T0_img
    zetaKdict['sigma'] = sigma
    
    pD_T0 = partialDual(T0, ZTSP_S, ZTTP)
    pD_T = partialDual(T, ZTSP_S, ZTTP)
    print('partial dual value at special solution with just T0', pD_T0, 'with full T', pD_T, 'norm of T0', la.norm(T0), 'norm of K', la.norm(K))
    return zetaK, pD_T, zetaKdict


def get_partialDual_matrices(zeta, ZTT, ZTS_S, E, S):
    ZTTP = ZTT + zeta*E
    ZTSP_S = ZTS_S + 0.5j*zeta*S
    T = la.solve(ZTTP, ZTSP_S)
    return T, ZTSP_S, ZTTP

def get_regular_constraintViolation(zeta, ZTT, ZTS_S, E, S):
    #evaluates the global cvx constraint violation at global multiplier value zeta
    #assumes that zeta>zetaK and so ZTTP is positive definite
    T, ZTSP_S, ZTTP = get_partialDual_matrices(zeta, ZTT, ZTS_S, E, S)
    
    return get_globalcvx_constraint(S, T, E)


def evaluate_nondefiniteZTT_regular_partialDual(zetaK, ZTT, ZTS_S, E, S, zeta_xtol=1e-6, zeta_rtol=1e-3):
    #for non-definite ZTT, returns the partial dual value at the regular solution zetaR>zetaK, optimal Tlist, and ZTTP
    eps0 = 1e-12
    zetaRcstrt_func = lambda zetaR: get_regular_constraintViolation(zetaR, ZTT, ZTS_S, E, S)
    #first check that the constraint violation at zetaK+eps0*zetaK is negative so that there is a regular solution
    zetaRcstrt_lb = zetaRcstrt_func(zetaK+eps0*zetaK)
    print('value of zetaRcstrt eps0*zetaK away from zetaK at', zetaK+eps0*zetaK, 'is', zetaRcstrt_lb)
    if zetaRcstrt_lb>=0: #eps0*zetaK to keep things relative
        return -1, -1, [],[] #indicate that no regular solution exists above zetaK+eps0
    
    zetaRmin = zetaK+eps0*zetaK
    deltazeta = 1.0
    while zetaRcstrt_func(zetaK+deltazeta)<0:
        zetaRmin = zetaK+deltazeta
        deltazeta *= 2
    
    zetaR_sol = root_scalar(zetaRcstrt_func, bracket=[zetaRmin, zetaK+deltazeta], xtol=zeta_xtol, rtol=zeta_rtol, method='brentq')
    zetaR = zetaR_sol.root
    print('status of root_scalar for finding zetaR', 'converged', zetaR_sol.converged, 'flag', zetaR_sol.flag)
    print('indefiniteZTT, zetaK is', zetaK, 'zetaR is', zetaR, 'with cvx constraint violation', zetaRcstrt_func(zetaR), flush=True)
    T, ZTSP_S, ZTTP = get_partialDual_matrices(zetaR, ZTT, ZTS_S, E, S)

    return zetaR, partialDual(T, ZTSP_S, ZTTP), T, ZTTP


def get_spatialProjopt_partialDual_singlematrix(Lags, grad, Hess, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, include=None, get_T=False, zeta_xtol=1e-6, zeta_rtol=1e-4):
    if include is None:
        include = [True]*len(Lags)

    get_grad = len(grad)>0
    get_Hess = len(Hess)>0
    
    Lagnum = len(Lags)
    ZTT = ZTTfunc(Lags)
    ZTS_S = ZTS_Sfunc(Lags, S)
    if get_grad or get_Hess:
        gradZTT = gradZTTfunc(Lags)
        gradZTS_S = gradZTS_Sfunc(Lags, S)
    
    
    flagRegular = True
    flagzeta0 = False
    
    zetaK, pDual_zetaK, zetaKdict = evaluate_nondefiniteZTT_kernel_solution(ZTT, ZTS_S, E, invsqrtE, S, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    T0_zetaK = zetaKdict['T0']
    K_zetaK = zetaKdict['K']
    ZTTP_zetaK = zetaKdict['ZTTP']
    
    if zetaK<0: # in this case ZTT is definite
        print('ZTT is definite')
        zeta0_T, zeta0_ZTSP_S, zeta0_ZTTP = get_partialDual_matrices(0, ZTT, ZTS_S, E, S)
        if get_globalcvx_constraint(S, zeta0_T, E)>=0:
            print('zeta=0 solution found', flush=True)
            flagzeta0 = True #needed when evaluating Hessian
            pDual = partialDual(zeta0_T, zeta0_ZTSP_S, zeta0_ZTTP)
            T = zeta0_T
            ZTTP = zeta0_ZTTP
        else: #solve for appropriate value of zetaR
            zetaRcstrt_func = lambda zetaR: get_regular_constraintViolation(zetaR, ZTT, ZTS_S, E, S)
            zetaRmin=0.0; zetaRmax = 1.0
            while zetaRcstrt_func(zetaRmax)<0:
                zetaRmin = zetaRmax
                zetaRmax *= 2
            
            zetaR_sol = root_scalar(zetaRcstrt_func, bracket=[zetaRmin, zetaRmax], xtol=zeta_xtol, rtol=zeta_rtol, method='brentq')
            zetaR = zetaR_sol.root
            print('zetaR is', zetaR, flush=True)
            T, ZTSP_S, ZTTP = get_partialDual_matrices(zetaR, ZTT, ZTS_S, E, S)
            pDual = partialDual(T, ZTSP_S, ZTTP)
    else: #ZTT is indefinite
        zetaR, pDual_zetaR, T_zetaR, ZTTP_zetaR = evaluate_nondefiniteZTT_regular_partialDual(zetaK, ZTT, ZTS_S, E, S, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)
        if pDual_zetaR<0:
            print('no regular solution found about zetaK, pDual_zetaK is', pDual_zetaK, flush=True)
            flagRegular = False
            pDual = pDual_zetaK
            T = T0_zetaK
            ZTTP = ZTTP_zetaK
        else:
            if pDual_zetaR<pDual_zetaK:
                print('found case where special solution is greater than regular solution.')
                print('pDual_zetaK', pDual_zetaK, 'pDual_zetaR', pDual_zetaR)
                print(Lags)
                #raise ValueError
            pDual = pDual_zetaR
            T = T_zetaR
            ZTTP = ZTTP_zetaR
    
    
    if get_grad:
        grad[:] = 0
        if flagRegular: #gradient is original gradient
            for i in range(len(Lags)):
                grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i]))
        else:
            unitw, unitv = la.eigh(invsqrtE @ ZTT @ invsqrtE)
            invsqrtEunitv = invsqrtE @ unitv[:,0]
            grad_zetaK = np.zeros(Lagnum)
            for i in range(Lagnum):
                grad_zetaK[i] = -np.real(np.vdot(invsqrtEunitv, gradZTT[i] @ invsqrtEunitv))
            
            globalcvxcstrt = np.imag(np.vdot(S,T)) - np.real(np.vdot(T, E @ T))
            for i in range(Lagnum):
                grad[i] += -np.real(np.vdot(T, gradZTT[i] @ T)) + 2*np.real(np.vdot(T, gradZTS_S[i])) + grad_zetaK[i] * globalcvxcstrt
    
    if get_Hess and flagRegular: #only compute Hessian when we are using a regular solution
        #first generate gradient of |T>
        gradT_Lags = []
        ZTTP_cho = la.cho_factor(ZTTP) #watch for possible floating point precision issues
        for i in range(len(gradZTT)):
            gradT_Lags.append(la.cho_solve(ZTTP_cho, -gradZTT[i] @ T + gradZTS_S[i]))
        
        if not flagzeta0: #in this case we need to compute pdv(zeta_R)
            gradT_zetaR = la.cho_solve(ZTTP_cho, -E @ T + 0.5j*S)
            gradcvx_Lags = np.zeros(Lagnum)
            for i in range(Lagnum):
                if include[i]:
                    gradcvx_Lags[i] = np.imag(np.vdot(S, gradT_Lags[i])) - 2*np.real(np.vdot(gradT_Lags[i], E @ T))
            gradcvx_zetaR = np.imag(np.vdot(S, gradT_zetaR)) - 2*np.real(np.vdot(gradT_zetaR, E @ T))
        
        Hess[:,:] = 0
        for i in range(Lagnum):
            if not include[i]:
                continue
            for j in range(i,Lagnum):
                if not include[j]:
                    continue
                Hess[i,j] += 2*np.real(np.vdot(gradT_Lags[i], -gradZTT[j] @ T + gradZTS_S[j]))
                if not flagzeta0: # in this case need extra contribution from implicit zetaR dependence
                    Hess[i,j] += -gradcvx_Lags[i]*gradcvx_Lags[j]/gradcvx_zetaR
                if i!=j:
                    Hess[j,i] = Hess[i,j]
                    
    if get_T:
        return pDual + dualconst, flagRegular, flagzeta0, T
    else:
        return pDual + dualconst, flagRegular, flagzeta0


def get_inc_spatialProjopt_partialDual_singlematrix(include, incLags, incgrad, incHess, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, get_T=False, zeta_xtol=1e-6, zeta_rtol=1e-4, printdual=True):
    get_grad = len(incgrad)>0
    get_Hess = len(incHess)>0

    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags
    
    if get_grad:
        grad = np.zeros(Lagnum)
    else:
        grad = np.array([])
    if get_Hess:
        Hess = np.zeros((Lagnum,Lagnum))
    else:
        Hess = np.array([])
        
    output = get_spatialProjopt_partialDual_singlematrix(Lags, grad, Hess, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include, get_T=get_T, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    if printdual:
        print('dual val is', output[0])
    if get_grad:
        incgrad[:] = grad[include]
    if get_Hess:
        incHess[:,:] = Hess[np.ix_(include,include)]
    return output



def get_spatialProjopt_kernel_partialDual_singlematrix(Lags, grad, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, include=None, get_T=False, zeta_xtol=1e-6, zeta_rtol=1e-4):
    if include is None:
        include = [True]*len(Lags)

    get_grad = len(grad)>0
    
    Lagnum = len(Lags)
    ZTT = ZTTfunc(Lags)
    ZTS_S = ZTS_Sfunc(Lags, S)
    
    if get_grad:
        gradZTT = gradZTTfunc(Lags)
        gradZTS_S = gradZTS_Sfunc(Lags, S)
        
    zetaK, pDual_zetaK, zetaKdict = evaluate_nondefiniteZTT_kernel_solution(ZTT, ZTS_S, E, invsqrtE, S, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    ZTTP = ZTT + zetaK * E
    ZTSP_S = ZTS_S + zetaK*1j*S/2.0
    eigw_ZTTP = zetaKdict['eigw_ZTTP']
    eigv_ZTTP = zetaKdict['eigv_ZTTP']
    img_mask = zetaKdict['img_mask']
    ker_mask = zetaKdict['ker_mask']
    T0 = zetaKdict['T0']
    K = zetaKdict['K']
    Ivecs = zetaKdict['Ivecs']
    Kvecs = zetaKdict['Kvecs']
    Pker = Kvecs @ Kvecs.conj().T
    sigma = zetaKdict['sigma']
    print('check satisfaction of global convex constraint', get_globalcvx_constraint(S, T0+K, E))
    if zetaK<0: # in this case ZTT is definite
        raise ValueError('ZTT is definite')
    
    if get_grad:
        grad[:] = 0
        #unpack relevant quantities for gradient calculation from zetaKdict
        T0 = zetaKdict['T0']
        
        unitw, unitv = la.eigh(invsqrtE @ ZTT @ invsqrtE)
        invsqrtEunitv = invsqrtE @ unitv[:,0]
        grad_zetaK = np.zeros(Lagnum)
        for i in range(Lagnum):
            grad_zetaK[i] = -np.real(np.vdot(invsqrtEunitv, gradZTT[i] @ invsqrtEunitv))
            
        globalcvxcstrt = np.imag(np.vdot(S,T0)) - np.real(np.vdot(T0, E @ T0))
        for i in range(Lagnum):
            grad[i] += -np.real(np.vdot(T0, gradZTT[i] @ T0)) + 2*np.real(np.vdot(T0, gradZTS_S[i])) + grad_zetaK[i] * globalcvxcstrt

        ######NOW EVALUATE 2Re[<K|ZTS_S>] part of gradient########
        #unpack necessary quantities from zetaKdict
        ZTTP = zetaKdict['ZTTP']
        K = zetaKdict['K']
        EKinv = zetaKdict['EKinv']
        EKinv_ZTSP_S = zetaKdict['EKinv_ZTSP_S']
        EKinv_S = zetaKdict['EKinv_S']
        EKinv_ET0 = zetaKdict['EKinv_ET0']
        sigma = zetaKdict['sigma']

        img_T0 = zetaKdict['img_T0']
        Ivecs = zetaKdict['Ivecs']
        img_ZTTP = Ivecs.conj().T @ ZTTP @ Ivecs # for evaluating gradT0
        img_E = Ivecs.conj().T @ E @ Ivecs # for evaluating gradT0
        gradT0list = []
        gradKlist = []
        gradsigmalist = []
        gradZTTPlist = []
        gradZTSP_Slist = []
        gradKvecslist = []
        gradPkerlist = []
        print('check T0', la.norm(T0 - Ivecs @ la.solve(img_ZTTP, Ivecs.conj().T @ ZTSP_S)))
        print('check T0 and img_T0', la.norm(T0 - (Ivecs @ img_T0)))
        for i in range(Lagnum):
            #evaluate gradT0
            gradZTTP = gradZTT[i] + grad_zetaK[i]*E
            gradZTSP_S = gradZTS_S[i] + grad_zetaK[i]*(1j*S/2.0)
            img_gradZTTP = Ivecs.conj().T @ gradZTTP @ Ivecs
            img_gradZTSP_S = Ivecs.conj().T @ gradZTSP_S
            gradT0 = Ivecs @ la.solve(img_ZTTP, -img_gradZTTP @ img_T0 + img_gradZTSP_S)

            #grad Kvecs
            gradKvecs = -(Ivecs.conj().T @ (gradZTTP @ Kvecs))
            print('shape of gradKvecs', gradKvecs.shape)
            print('shape of eigw_ZTTP[img_mask]', eigw_ZTTP[img_mask].shape)
            gradKvecs = (gradKvecs.T / eigw_ZTTP[img_mask]).T
            gradKvecs = Ivecs @ gradKvecs
            print('shape of gradKvecs', gradKvecs.shape)
            gradKvecslist.append(gradKvecs)
            #grad Pker
            gradPker = gradKvecs @ Kvecs.conj().T + Kvecs @ gradKvecs.conj().T
            gradPkerlist.append(gradPker)
            #evaluate alpha for calculating gradK
            vl = (2*E@(T0+K)) - 1j*S
            vr = EKinv @ (gradZTSP_S/sigma - (E@gradT0))
            LHSfactor = np.real( np.vdot(vl, EKinv_ZTSP_S) )
            RHSfactor = np.real( np.vdot(vl, gradT0 + vr) )
            alpha = RHSfactor / LHSfactor

            gradK = -alpha*EKinv_ZTSP_S + vr

            grad[i] += 2*np.real(np.vdot(gradK,ZTSP_S) + np.vdot(K,gradZTSP_S)) #finally
            gradT0list.append(gradT0)
            gradKlist.append(gradK)
            gradsigmalist.append(sigma**2 * alpha)
            gradZTTPlist.append(gradZTTP)
            gradZTSP_Slist.append(gradZTSP_S)
        
    if get_T and get_grad:
        return pDual_zetaK + dualconst, zetaK, Pker, Kvecs, T0, K, ZTTP, ZTSP_S, sigma, grad_zetaK, gradPkerlist, gradKvecslist, gradT0list, gradKlist, gradZTTPlist, gradZTSP_Slist, gradsigmalist
    elif get_T:
        return pDual_zetaK + dualconst, zetaK, Pker, Kvecs, T0, K, ZTTP, ZTSP_S, sigma
    else:
        return pDual_zetaK + dualconst


def get_inc_spatialProjopt_kernel_partialDual_singlematrix(include, incLags, incgrad, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=0.0, get_T=False, zeta_xtol=1e-6, zeta_rtol=1e-4, printdual=True):
    get_grad = len(incgrad)>0

    Lagnum = len(include)
    Lags = np.zeros(Lagnum)
    Lags[include] = incLags
    
    if get_grad:
        grad = np.zeros(Lagnum)
    else:
        grad = np.array([])
        
    output = get_spatialProjopt_kernel_partialDual_singlematrix(Lags, grad, S, E, invsqrtE, ZTTfunc, gradZTTfunc, ZTS_Sfunc, gradZTS_Sfunc, dualconst=dualconst, include=include, get_T=get_T, zeta_xtol=zeta_xtol, zeta_rtol=zeta_rtol)

    if get_T:
        dualval = output[0]
    else:
        dualval = output
    if printdual:
        print('dual val is', dualval)
    if get_grad:
        incgrad[:] = grad[include]
    return output
