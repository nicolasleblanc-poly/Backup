import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import nlopt
import time
from .spatialProjopt_Zops_Msparse import get_ZTT, get_ZTT_mineig_grad


feasiblept = None


def check_ZTT_mineig_grad(incLags, incgrad, include, O, gradZTT, pos_tol=1e-4):
    Lags = np.zeros(len(include))
    Lags[include] = incLags
    ZTT = get_ZTT(Lags, O, gradZTT)
    #check time taken for eig comp.
    t = time.time()
    eigw, eigv = spla.eigsh(ZTT, k=1, which='SA', return_eigenvectors=True)
    print('mineig', eigw[0], 'time taken for eig computation', time.time()-t, flush=True)
    
    if eigw[0]>pos_tol:
        global feasiblept
        feasiblept = incLags
        raise ValueError('Found feasible point.')
    
    if len(incgrad)>0:
        inc_ind = 0
        for i in range(len(gradZTT)):
            if include[i]:
                incgrad[inc_ind] = np.real(np.vdot(eigv[:,0], gradZTT[i].dot(eigv[:,0])))
                inc_ind += 1

    return eigw[0]


def spatialProjopt_find_feasiblept_NLOPT(Lagnum, include, zeta_mask, O, gradZTT, pos_tol=1e-4):
    incLagnum = int(np.sum(include))
    initincLags = np.zeros(incLagnum)
    inc_zeta_mask = zeta_mask[include] #the included multipliers corresponding to definite constraints
    initincLags[inc_zeta_mask] = 1.0

    mineigfunc = lambda dof, grad: check_ZTT_mineig_grad(dof, grad, include, O, gradZTT, pos_tol=pos_tol)
    
    opt = nlopt.opt(nlopt.LD_LBFGS, incLagnum)
    opt.set_max_objective(mineigfunc)

    lb = -np.inf * np.ones(incLagnum)
    lb[inc_zeta_mask] = 0
    rb = np.inf * np.ones(incLagnum)
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(rb)

    opt.set_ftol_rel(1e-8)
    try:
        optLags = opt.optimize(initincLags)
    except ValueError:
        global feasiblept
        print('found dual feasible starting point')
        Lags = np.zeros(Lagnum)
        Lags[include] = feasiblept
        return Lags

    print('did not find feasible point via optimization, returning found best starting point')
    Lags = np.zeros(Lagnum)
    Lags[include] = optLags
    Lags[zeta_mask] = np.abs(Lags[zeta_mask]) + pos_tol
    return Lags
