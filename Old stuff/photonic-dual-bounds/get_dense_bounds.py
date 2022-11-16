import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import sys, time
sys.path.append('/u/pengning/Photonic_Dual_Bounds/photonic-dual-bounds/')



from dualbound.Lagrangian.spatialProjopt_multiSource_Zops_singlematrix_numpy import get_gradZTT, get_gradZTS_S, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity, get_inc_ZTT_mineig

from dualbound.Lagrangian.spatialProjopt_multiSource_dualgradHess_fakeS_singlematrix_numpy import get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix

from dualbound.Lagrangian.spatialProjopt_multiSource_feasiblept_singlematrix import spatialProjopt_find_feasiblept

from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix

from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart


def get_multiSource_dense_bound(n_S, Si, O_lin, O_quad, Plist, UPlist, include, dualconst=0.0, initLags=None, getT=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, alg='Newton'):

    gradZTT = get_gradZTT(n_S, UPlist)
    gradZTS_S = get_gradZTS_S(n_S, Si, Plist)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(n_S, dof, include, O_quad, gradZTT)

    mineigfunc = lambda dof: get_inc_ZTT_mineig(n_S, dof, include, O_quad, gradZTT, eigvals_only=False)

    if initLags is None:
        Lags = spatialProjopt_find_feasiblept(n_S, len(include), include, O_quad, gradZTT)
    else:
        Lags = initLags.copy()
    print('Lags', Lags)

    while True:
        tmp = check_spatialProj_Lags_validity(n_S, Lags, O_quad, gradZTT)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[1] *= 1.5

    if alg=='Newton':
        dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(n_S, dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)
        optincLags, optincgrad, dualval, objval = fakeS_with_restart_singlematrix(Lags[include], dgHfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period)
    elif alg=='LBFGS':
        dgfunc = lambda dof, dofgrad, fSl, get_grad=True: get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(n_S, dof, dofgrad, [], include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=False)
        optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period)
    else:
        raise ValueError('alg has to be either Newton or LBFGS')
        
    optLags = np.zeros(len(include), dtype=np.double)
    optLags[include] = optincLags[:]
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]

    return optLags, optgrad, dualval, objval
