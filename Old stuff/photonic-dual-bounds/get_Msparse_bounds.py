import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

import sys, time
sys.path.append('C:/Users/Nic.Leblanc/Documents/GitHub/Extracted-power-optimization/photonic-dual-bounds') #load package directory into system path so the script can access the package


from dualbound.Lagrangian.spatialProjopt_Zops_Msparse import Cholesky_analyze_ZTT, get_multiSource_Msparse_gradZTS_S, get_Msparse_gradZTS_S, get_multiSource_Msparse_gradZTT, get_Msparse_gradZTT, get_Msparse_inc_ZTT_mineig, get_Msparse_inc_PD_ZTT_mineig, check_Msparse_spatialProj_Lags_validity, check_Msparse_spatialProj_incLags_validity

from dualbound.Lagrangian.spatialProjopt_vecs_Msparse import get_ZTTcho_Tvec

from dualbound.Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgrad_fakeS_Msparse, get_inc_spatialProj_dualgradHess_fakeS_Msparse

from dualbound.Lagrangian.spatialProjopt_feasiblept_Msparse import spatialProjopt_find_feasiblept 

from dualbound.Optimization.BFGS_fakeSource_with_restart import BFGS_fakeS_with_restart

from dualbound.Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix


def get_Msparse_bound(Si, O_lin, O_quad, GinvdagPdaglist, UPlist, include, dualconst=0.0, initLags=None, getT=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, alg='Newton'):
    """
    The matrices in GinvdagPdaglist and UPlist are all sparse
    primal degrees of freedom are components of G @ T @ S
    """
    gradZTT = get_multiSource_Msparse_gradZTT(1, UPlist)
    print('len gradZTT', len(gradZTT))
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
    gradZTS_S = get_multiSource_Msparse_gradZTS_S(1, Si, GinvdagPdaglist)

    validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)

    #mineigfunc = lambda dof: get_Msparse_inc_ZTT_mineig(dof, include, O_quad, UPlist, eigvals_only=False)
    mineigfunc = lambda dof: get_Msparse_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    if initLags is None:
        #Lags = spatialProjopt_find_feasiblept(len(include), include, O_quad, gradZTT)
        Lags = np.random.rand(len(include))
        Lags[np.logical_not(include)] = 0.0
        Lags[1] = np.abs(Lags[1])+0.01
        #Lags[1] = -10.0
    else:
        Lags = initLags.copy()

    print('Lags', Lags)
    while True:
        tmp = check_Msparse_spatialProj_Lags_validity(Lags, O_quad, gradZTT)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[1] *= 1.5


    if alg=='Newton':
        dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fakeS_Msparse(dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess, chofac=ZTTchofac)
        optincLags, optincgrad, dualval, objval = fakeS_with_restart_singlematrix(Lags[include], dgHfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period)
    elif alg=='LBFGS':
        dgfunc = lambda dof, dofgrad, fSl, get_grad=True: get_inc_spatialProj_dualgrad_fakeS_Msparse(dof, dofgrad, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, chofac=ZTTchofac)
        optincLags, optincgrad, dualval, objval = BFGS_fakeS_with_restart(Lags[include], dgfunc, validityfunc, mineigfunc, opttol=opttol, fakeSratio=fakeSratio, reductFactor=reductFactor, iter_period=iter_period)
    else:
        raise ValueError('alg must be Newton or LBFGS')
    
    optLags = np.zeros(len(include), dtype=np.double)
    optLags[include] = optincLags[:]
    optgrad = np.zeros(len(include), dtype=np.double)
    optgrad[include] = optincgrad[:]

    print('the remaining constraint violations')
    print(optgrad)

    if getT:
        ZTS_S = O_lin.copy()
        ZTT = O_quad.copy()
        for i in range(len(optLags)):
            ZTS_S += optLags[i] * gradZTS_S[i]
            ZTT += optLags[i] * gradZTT[i]
        _, optT = get_ZTTcho_Tvec(ZTT, ZTS_S, ZTTchofac)
        return optLags, optgrad, dualval, objval, optT
    
    else:
        return optLags, optgrad, dualval, objval


