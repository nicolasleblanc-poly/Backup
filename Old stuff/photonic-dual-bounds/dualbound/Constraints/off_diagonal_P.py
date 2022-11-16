import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .mask_iterative_splitting import reduce_Plist

from ..Lagrangian.spatialProjopt_multiSource_Zops_singlematrix_numpy import get_gradZTT, get_gradZTS_S, check_spatialProj_Lags_validity, check_spatialProj_incLags_validity, get_inc_ZTT_mineig

from ..Lagrangian.spatialProjopt_multiSource_dualgradHess_fakeS_singlematrix_numpy import get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix

from ..Lagrangian.spatialProjopt_multiSource_feasiblept_singlematrix import spatialProjopt_find_feasiblept

from ..Lagrangian.spatialProjopt_Zops_Msparse import get_ZTT, get_Msparse_gradZTT, get_Msparse_gradZTS_S, Cholesky_analyze_ZTT, check_Msparse_spatialProj_Lags_validity, check_Msparse_spatialProj_incLags_validity, get_Msparse_inc_PD_ZTT_mineig

from ..Lagrangian.spatialProjopt_dualgradHess_fakeS_Msparse import get_inc_spatialProj_dualgradHess_fakeS_Msparse

from ..Optimization.fakeSource_with_restart_singlematrix import fakeS_with_restart_singlematrix, fakeS_single_restart_Newton


"""
two methods for brute force iterative splitting of off-diagonal constraints
"""
def split_off_diagonal_Plist(offd_Plist, Lags):
    split_offd_Plist = []
    split_Lags = []
    flag = False # indicates whether any region has been split or not
    Pdofnum = len(offd_Plist[0])
    Pdof_inds = np.arange(Pdofnum)
    for i in range(len(offd_Plist)):
        if np.sum(offd_Plist)<=1:
            split_offd_Plist.append(offd_Plist[i])
            split_Lags.append(Lags[2*i])
            split_Lags.append(Lags[2*i+1])
        else:
            flag = True
            old_inds = Pdof_inds[offd_Plist[i]]
            new_offd_P = np.zeros(Pdofnum, dtype=np.bool)
            new_offd_P[old_inds[old_inds <= (old_inds[0]+old_inds[-1])//2]] = True
            split_offd_Plist.append(new_offd_P)

            new_offd_P = np.zeros(Pdofnum, dtype=np.bool)
            new_offd_P[old_inds[old_inds > (old_inds[0]+old_inds[-1])//2]] = True
            split_offd_Plist.append(new_offd_P)

            split_Lags.extend([Lags[2*i], Lags[2*i+1]] * 2)

    if not flag:
        raise ValueError('no more subdivisions possible')

    return split_offd_Plist, np.array(split_Lags)



def off_diagonal_P_iterative_splitting(U, primaldofnum, dualoptFunc, outputFunc=None):

    Pdofnum = primaldofnum ** 2
    #store representations of off-diagonal P as a 1D boolean array

    diag_global_Pmat = np.eye(primaldofnum, dtype=np.complex)

    offd_Plist = [np.ones(Pdofnum, dtype=np.bool)]
    initLags = np.array([0.0, 1.0, 0.0, 0.0]) #initial guess to start dual optimization, positive multiplier for global Asym constraint
    
    while True: #do iterative splitting of offd_Plist, and always combine with the original global constraints
        print('total number of general off-diagonal P constraints', len(offd_Plist))
        #generate Plist and UPlist for dualoptFunc

        Pmatlist = [diag_global_Pmat]
        UPlist = [U]
        for i in range(len(offd_Plist)):
            Pmat = np.reshape(offd_Plist[i].astype(np.complex), (primaldofnum, primaldofnum))
            Pmatlist.append(Pmat)
            UPlist.append(U @ Pmat)

        include = np.ones_like(initLags, dtype=np.bool)
        optLags, optgrad, dual, obj = dualoptFunc(initLags, Pmatlist, UPlist, include)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, dual, obj)

        #do splitting
        offd_Plist, offd_Lags = split_off_diagonal_Plist(offd_Plist, optLags[2:])
        initLags = np.zeros(len(offd_Lags)+2)
        initLags[:2] = optLags[:2] #the global multipliers
        initLags[2:] = offd_Lags[:] #the general constraint multipliers



def dual_space_reduction_iteration_dense_align_mineig_maxviol(chi, Si, U, O_lin, O_quad, Pstruct=None, P0phase=1.0+0j, Pnum=1, dualconst=0.0, opttol=1e-2, fakeSratio=1e-2, gradConverge=False, singlefS=False, iter_period=20, outputFunc=None):
    """
    dual space reduction for dense off-diagonal P constraints
    Pstruct is a boolean correlation matrix indicating structure of P
    Pstruct[i,j] = True if P can have a non-zero (i,j) entry and False otherwise
    generate new projection constraints to both have large Lagrangian gradient
    and have the Lagrangian gradient point in a direction that increases the minimum eigenvalue of ZTT
    """

    if Pstruct is None:
        Pstruct = np.ones_like(U, dtype=np.bool)
    
    P0 = np.eye(U.shape[0], dtype=np.complex) * P0phase
    P0 *= Pstruct #enforcing structure of P, though generally we would expect the diagonal of Pstruct to be all true
    P0 /= la.norm(P0)
    
    Plist = [P0]
    UPlist = [U @ P0]

    gradZTT = get_gradZTT(1, UPlist)
    gradZTS_S = get_gradZTS_S(1, Si, Plist)

    include = np.ones(len(gradZTT), dtype=np.bool)

    validityfunc = lambda dof: check_spatialProj_incLags_validity(1, dof, include, O_quad, gradZTT)

    mineigfunc = lambda dof: get_inc_ZTT_mineig(1, dof, include, O_quad, gradZTT, eigvals_only=False)

    dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(1, dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)


    Lags = np.zeros(len(gradZTT))
    Lags[1] = 1.0

    while True:
        tmp = check_spatialProj_Lags_validity(1, Lags, O_quad, gradZTT)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[1] *= 1.5

    fSlist = []
    optLags = Lags[include]

    iternum = 0
    mindual = np.inf
    minLags = mingrad = minobj = None
    
    while True: #perhaps put in a convergence criterion later
        iternum += 1

        optLags, optgrad, optdual, optobj, fSlist = fakeS_single_restart_Newton(optLags, dgHfunc, validityfunc, mineigfunc, fSlist=fSlist, opttol=opttol, gradConverge=gradConverge, singlefS=singlefS, fakeSratio=fakeSratio, iter_period=iter_period)

        print('at dimension reduction iteration #', iternum, 'optdual and norm(optgrad) are', optdual, la.norm(optgrad))
        if optdual<mindual:
            mindual = optdual
            minobj = optobj
            minLags = optLags.copy()
            mingrad = optgrad.copy()
        print('the tightest dual bound found so far is', mindual)
        print('number of projection constraints', len(Plist))
        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj)
            print('minimum found so far')
            outputFunc(minLags, mingrad, mindual, minobj)
        
        ####adjust list of projection constraints
        ZTT = O_quad.copy()
        ZTS_S = O_lin.copy()
        for i in range(len(optLags)):
            ZTT += optLags[i] * gradZTT[i]
            ZTS_S += optLags[i] * gradZTS_S[i]

        if len(Plist)-1>Pnum: #reduce Plist down to global constraint + one general constraint
            print('reduce Plist')
            print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
            
            Plist_new, optLags_new = reduce_Plist(Plist, optLags) #dual space reduction
            while len(Plist_new)>2:
                Plist_new, optLags_new = reduce_Plist(Plist_new, optLags_new)


            optLags = optLags_new
            Plist = Plist_new
            UPlist = []
            for i in range(len(Plist)):
                UPlist.append(U @ Plist[i])
            gradZTT = get_gradZTT(1, UPlist)
            gradZTS_S = get_gradZTS_S(1, Si, Plist)

        ##################get new projection constraint#####################
        ######
        optT = la.solve(ZTT, ZTS_S)
        UdagT = U.conj().T @ optT
        violation = np.outer(np.conj(Si), optT) - np.outer(np.conj(UdagT), optT) #off-diagonal P, violation generalized from point-wise multiplication to outer product

        for i in range(len(fSlist)): #contribution to violation from fake Source terms
            ZTTinvfS = la.solve(ZTT, fSlist[i])
            UdagZTTinvfS = U.conj().T @ ZTTinvfS
            violation -= np.outer(np.conj(UdagZTTinvfS), ZTTinvfS)

        ###evaluate direction for increasing mineig of ZTT
        eigw, eigv = la.eigh(ZTT)
        eigw = eigw[0]
        eigv = eigv[:,0]

        mineigfac = np.outer(np.conj(U.conj().T @ eigv), eigv)

        mineigfac_phase = mineigfac / np.abs(mineigfac)
        Laggradfac_phase = -violation / np.abs(violation)

        P_new = np.conj(mineigfac_phase + Laggradfac_phase)
        #P_new *= np.abs(np.real(violation)) #scale with violation size like before
        P_new *= np.abs(np.real(-P_new * violation)) #scale with real part of Lagrangian gradient after rotation in phase
        
        
        P_new[np.isnan(P_new)] = 0.0
        P_new *= Pstruct #enforce structure

        ###check that SymUP_new has a positive mineig ZTT gradient
        UP_new = U @ P_new
        print('check SymUP_new ZTT mineig gradient', np.real(np.vdot(eigv, UP_new @ eigv)))
        
        print('norm of new projection matrix before orthogonalization', la.norm(P_new))

        #orthogonalize against prior projections
        for i in range(len(Plist)):
            P_new -= np.vdot(Plist[i].flatten(), P_new.flatten()) * Plist[i]

        print('norm of new projection matrix after orthogonalization', la.norm(P_new))
        
        P_new /= la.norm(P_new) #normalize
        UP_new = U @ P_new
        #update the dual optimization parameters
        Plist.append(P_new)
        UPlist.append(UP_new)
        
        optLags_new = np.zeros(len(optLags)+2)
        optLags_new[:-2] = optLags[:]
        optLags = optLags_new

        gradZTT.extend([(UP_new+UP_new.conj().T)/2, (UP_new-UP_new.conj().T)/2j])
        gradZTS_S.append(P_new.conj().T @ Si / 2)
        gradZTS_S.append(1j*gradZTS_S[-1])

        include = np.ones(len(gradZTT), dtype=np.bool)

        validityfunc = lambda dof: check_spatialProj_incLags_validity(1, dof, include, O_quad, gradZTT)

        mineigfunc = lambda dof: get_inc_ZTT_mineig(1, dof, include, O_quad, gradZTT, eigvals_only=False)

        dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_multiSource_dualgradHess_fakeS_singlematrix(1, dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)



def outer_sparse_Pstruct(a, b, Pstruct):
    """
    equivalent to np.outer(a,b) * Pstruct where Pstruct is a boolean mask
    here Pstruct is represented as a sparse coo array with 1s on supported entries
    """
    outer_data = a[Pstruct.row] * b[Pstruct.col]
    return outer_data


import time

def dual_space_reduction_iteration_Msparse_align_mineig_maxviol(chi, Si, Ginv, O_lin, O_quad, Pstruct=None, P0phase=1.0+0j, Pnum=1, dualconst=0.0, opttol=1e-2, fakeSratio=1e-2, gradConverge=False, singlefS=False, iter_period=20, outputFunc=None):
    """
    dual space reduction for sparse off-diagonal P constraints
    Pstruct is a sparse correlation matrix indicating structure of P
    Pstruct[i,j] = 1 if P can have a non-zero (i,j) entry and 0 otherwise
    generate new projection constraints to both have large Lagrangian gradient
    and have the Lagrangian gradient point in a direction that increases the minimum eigenvalue of ZTT
    """

    if Pstruct is None: #for sparse off-diagonal P, default assume off-diagonals are 0
        Pstruct = sp.eye(Ginv.shape[0], format='coo')
    Pstruct = Pstruct.tocoo() #for use later with outer_sparse_Pstruct

    P0 = sp.eye(Ginv.shape[0], dtype=np.complex, format='csc') * P0phase
    P0 /= spla.norm(P0)

    P0_data = np.zeros_like(Pstruct.row, dtype=np.complex)
    for i in range(len(Pstruct.row)):
        P0_data[i] = P0[Pstruct.row[i],Pstruct.col[i]]
    ###TODO: re-write this so that Plist actually contains 1D arrays that can be viewed as the data part of a COO sparse P matrix, this representation makes it easy to orthogonalize new projection proposals
    Pdatalist = [P0_data]
    GinvdagPdag = Ginv.conj().T @ P0.conj().T
    GinvdagPdaglist = [GinvdagPdag]
    UPlist = [(Ginv.conj().T @ GinvdagPdag.conj().T)/np.conj(chi) - GinvdagPdag.conj().T]

    print('check AsymUM definiteness')
    t = time.time()
    UM = UPlist[0].todense()
    AsymUM = (UM-UM.conj().T)/2j
    print(la.eigvalsh(AsymUM)[0])
    print('time taken to evaluate eigenvalues of dense AsymUM:', time.time()-t, flush=True)

    AsymUM_sp = (UPlist[0] - UPlist[0].conj().T)/2j
    t = time.time()
    eig0_AsymUM = spla.eigsh(AsymUM_sp, k=1, sigma=0.0, which='LM', return_eigenvectors=False)
    print('using sparse ARPACK method, mineig for AsymUM is', eig0_AsymUM)
    print('time used', time.time()-t, flush=True)
    
    gradZTT = get_Msparse_gradZTT(UPlist)
    gradZTS_S = get_Msparse_gradZTS_S(Si, GinvdagPdaglist)
    ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
    
    include = np.ones(len(gradZTT), dtype=np.bool)

    validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)
    
    mineigfunc = lambda dof: get_Msparse_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    
    dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fakeS_Msparse(dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)


    Lags = np.zeros(len(gradZTT))

    Lags[1] = 1.0

    while True:
        tmp = check_Msparse_spatialProj_Lags_validity(Lags, O_quad, gradZTT, chofac=ZTTchofac)
        if tmp>0:
            break
        print(tmp, flush=True)
        print('zeta', Lags[1])
        Lags[1] *= 1.5

    fSlist = []
    optLags = Lags[include]

    iternum = 0
    mindual = np.inf
    minLags = mingrad = minobj = None
    
    while True: #perhaps put in a convergence criterion later
        iternum += 1

        optLags, optgrad, optdual, optobj, fSlist = fakeS_single_restart_Newton(optLags, dgHfunc, validityfunc, mineigfunc, fSlist=fSlist, opttol=opttol, gradConverge=gradConverge, singlefS=singlefS, fakeSratio=fakeSratio, iter_period=iter_period)

        print('at dimension reduction iteration #', iternum, 'optdual and norm(optgrad) are', optdual, la.norm(optgrad))
        if optdual<mindual:
            mindual = optdual
            minobj = optobj
            minLags = optLags.copy()
            mingrad = optgrad.copy()
        print('the tightest dual bound found so far is', mindual)
        print('number of projection constraints', len(Pdatalist))
        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj)
            print('minimum found so far')
            outputFunc(minLags, mingrad, mindual, minobj)

        ####adjust list of projection constraints
        ZTT = O_quad.copy()
        ZTS_S = O_lin.copy()
        for i in range(len(optLags)):
            ZTT += optLags[i] * gradZTT[i]
            ZTS_S += optLags[i] * gradZTS_S[i]

        if len(Pdatalist)-1>Pnum: #reduce Plist down to global constraint + one general constraint
            print('reduce Pdatalist')
            print('len optLags', len(optLags), 'len gradZTT, gradZTS_S', len(gradZTT), len(gradZTS_S))
            
            Pdatalist_new, optLags_new = reduce_Plist(Pdatalist, optLags) #dual space reduction
            while len(Pdatalist_new)>2:
                Pdatalist_new, optLags_new = reduce_Plist(Pdatalist_new, optLags_new)

            optLags = optLags_new
            Pdatalist = Pdatalist_new
            GinvdagPdaglist = []
            UPlist = []
            for i in range(len(Pdatalist)):
                P = sp.coo_matrix((Pdatalist[i], (Pstruct.row,Pstruct.col))).tocsc()
                GinvdagPdag = Ginv.conj().T @ P.conj().T
                GinvdagPdaglist.append(GinvdagPdag)
                UPlist.append(Ginv.conj().T @ GinvdagPdag.conj().T / np.conj(chi) - GinvdagPdag.conj().T)

            gradZTT = get_Msparse_gradZTT(UPlist)
            gradZTS_S = get_Msparse_gradZTS_S(Si, GinvdagPdaglist)

        ##################get new projection constraint#####################
        optGT = spla.spsolve(ZTT, ZTS_S)
        optT = Ginv @ optGT
        UdagT = optT/chi - optGT
        
        violation = outer_sparse_Pstruct(np.conj(Si), optT, Pstruct) - outer_sparse_Pstruct(np.conj(UdagT), optT, Pstruct) #off-diagonal P, violation generalized from point-wise multiplication to outer product

        for i in range(len(fSlist)): #contribution to violation from fake Source terms
            ZTTinvfS = spla.spsolve(ZTT, fSlist[i])
            GinvZTTinvfS = Ginv @ ZTTinvfS
            violation -= (1.0/np.conj(chi)) * outer_sparse_Pstruct(np.conj(GinvZTTinvfS), GinvZTTinvfS, Pstruct) - outer_sparse_Pstruct(np.conj(ZTTinvfS), GinvZTTinvfS, Pstruct)

        ###evaluate direction for increasing mineig of ZTT
        try:
            eigw, eigv = spla.eigsh(ZTT, k=1, sigma=0.0, which='LM', return_eigenvectors=True)
        except BaseException as err:
            print('encountered error in sparse eigenvalue evaluation', err)
            eigw, eigv = la.eigh(ZTT.toarray())

        eigw = eigw[0]
        eigv = eigv[:,0]

        Ginveigv = Ginv @ eigv
        mineigfac = outer_sparse_Pstruct(np.conj(Ginveigv), Ginveigv, Pstruct)/np.conj(chi) - outer_sparse_Pstruct(np.conj(eigv), Ginveigv, Pstruct)

        
        ###set the entries of the new projection matrix Pij so we align the gradients of mineig ZTT and the Sym(UP) constraint

        mineigfac_phase = mineigfac / (np.abs(mineigfac))
        Laggradfac_phase = -violation / np.abs(violation) #minus sign since we are doing dual minimization

        Pdata_new = np.conj(mineigfac_phase + Laggradfac_phase)
        Pdata_new *= np.abs(np.real(-Pdata_new * violation)) #scale with real part of Lagrangian gradient after rotation in phase

        print('norm of new projection matrix before orthogonalization', la.norm(Pdata_new))

        #orthogonalize against prior projections
        for i in range(len(Pdatalist)):
            Pdata_new -= np.vdot(Pdatalist[i], Pdata_new) * Pdatalist[i]

        print('norm of new projection matrix after orthogonalization', la.norm(Pdata_new))

        #update all of the lists and dofs for next iteration round
        Pdata_new /= la.norm(Pdata_new) #normalize
        P_new = sp.coo_matrix((Pdata_new, (Pstruct.row,Pstruct.col))).tocsc()
        GinvdagPdag = Ginv.conj().T @ P_new.conj().T
        UP_new = Ginv.conj().T @ GinvdagPdag.conj().T / np.conj(chi) - GinvdagPdag.conj().T

    
        #update the dual optimization parameters
        Pdatalist.append(Pdata_new)
        GinvdagPdaglist.append(GinvdagPdag)
        UPlist.append(UP_new)

        gradZTT.extend([(UP_new+UP_new.conj().T)/2, (UP_new-UP_new.conj().T)/2j])
        gradZTS_S.append(GinvdagPdag @ Si / 2)
        gradZTS_S.append(1j*gradZTS_S[-1])

        include = np.ones(len(gradZTT), dtype=np.bool)
        
        optLags_new = np.zeros(len(optLags)+2)
        optLags_new[:-2] = optLags[:]
        optLags = optLags_new

        ZTTchofac = Cholesky_analyze_ZTT(O_quad, gradZTT)
        validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=ZTTchofac)
        #validityfunc = lambda dof: check_Msparse_spatialProj_incLags_validity(dof, include, O_quad, gradZTT, chofac=None)
    
        mineigfunc = lambda dof: get_Msparse_inc_PD_ZTT_mineig(dof, include, O_quad, gradZTT, eigvals_only=False)

    
        dgHfunc = lambda dof, dofgrad, dofHess, fSl, get_grad=True, get_Hess=True: get_inc_spatialProj_dualgradHess_fakeS_Msparse(dof, dofgrad, dofHess, include, O_lin, O_quad, gradZTS_S, gradZTT, fSl, dualconst=dualconst, get_grad=get_grad, get_Hess=get_Hess)

        
