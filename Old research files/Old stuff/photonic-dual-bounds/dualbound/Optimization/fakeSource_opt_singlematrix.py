#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:57:05 2020

@author: pengning
"""

import numpy as np
import time
    

def fakeS_opt_singlematrix(initdof, S1, dgHfunc, validityfunc, mineigfunc, opttol=1e-2, fakeSratio=1e-2, check_iter_period=20):
    """
    does fake source dual optimization
    dgHfunc(dof, dofgrad, dofHess, S1, fSlist) returns the value of the dual+\sum <fS_i|ZTT^-1|fS_i> given dof, and stores the gradient and Hessian in dofgrad and dofHess
    validityfunc(dof) returns 1 if dof is within domain of duality and -1 otherwise
    fSlist starts off empty and the algorithm adds in fake source terms as necessary
    """
    
    fSlist = []
    
    dofnum = len(initdof)
    dof = initdof.copy() #.copy() because we don't wont to modify the initialization array in place
    dofgrad = np.zeros(dofnum)
    dofHess = np.zeros((dofnum,dofnum))
    
    tic = time.time()
    
    iternum = 0
    prevD = np.inf
    alphaopt_grad=1.0
    alphaopt_Hess=1.0
    tol_orthoS = 1e-1
    dualfunc = lambda d: dgHfunc(d, [],[], S1, fSlist, get_grad=False, get_Hess=False)
    
    while True:
        iternum += 1
        
        print('the iteration number is:', iternum, flush=True)
        
        doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
        
        dualval = dgHfunc(dof, dofgrad, dofHess, S1, fSlist, get_grad=True, get_Hess=(not doGD))
        objval = dualval - dof @ dofgrad
        abs_cstrt_sum = np.abs(dof) @ np.abs(dofgrad)
        print('current dual, objective, absolute sum of constraint violation are', dualval, objval, abs_cstrt_sum)
        
        if np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)): #objective convergence termination
            break
        
        if iternum % check_iter_period == 0:
            print('previous dual is', prevD)
            if np.abs(prevD-dualval)<np.abs(dualval)*1e-3: #dual convergence / stuck optimization termination
                break
            prevD = dualval
        
        normgrad = np.linalg.norm(dofgrad)
        if not doGD:
            Ndir = np.linalg.solve(dofHess, -dofgrad)
            normNdir = np.linalg.norm(Ndir)
            pdir = Ndir / normNdir
            print('do regular Hessian step')
            print('normNdir is', normNdir)
            print('normgrad is', normgrad)
            print('Ndir dot grad is', np.dot(Ndir, dofgrad))
        if doGD:
            print('do regular gradient step')
            pdir = -dofgrad/normgrad
            print('normgrad is', normgrad)
            
        c1 = 0.5; c2 = 0.7 #the parameters for doing line search
        if doGD:
            alpha_start = alphaopt_grad
        else:
            alpha_start = alphaopt_Hess
        alpha = alpha_start
        
        print('alpha before feasibility backtrack', alpha)
        while validityfunc(dof+alpha*pdir)<=0:
            alpha *= c2
        
        alpha_feas = alpha
        print('alpha before backtracking is', alpha_feas)
        alphaopt = alpha
        Dopt = np.inf
        while True:
            tmp = dualfunc(dof+alpha*pdir)
            if tmp<Dopt: #the dual is still decreasing as we backtrack, continue
                Dopt = tmp; alphaopt=alpha
            else:
                alphaopt=alpha
                break
            if tmp<=dualval + c1*alpha*(pdir @ dofgrad): #Armijo backtracking condition
                alphaopt = alpha
                break
            alpha *= c2
        
        if alphaopt/alpha_start>(c2+1)/2: #in this case can start with bigger step
            alpha_newstart = alphaopt*2
        else:
            alpha_newstart = alphaopt
            if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                print('encountered feasibility wall, adding a fake source term')
                singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                mineigw, mineigv = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                
                fakeSval = dgHfunc(dof, [], [], np.zeros_like(S1), [mineigv], get_grad=False, get_Hess=False)

                epsS = np.sqrt(fakeSratio*np.abs(dualval/fakeSval))
                fSlist.append(epsS * mineigv) #add new fakeS to fSlist
                print('length of fSlist', len(fSlist))
        
        print('stepsize alphaopt is', alphaopt, '\n')
        dof += alpha*pdir
        if doGD:
            alphaopt_grad = alpha_newstart
        else:
            alphaopt_Hess = alpha_newstart
        """
        if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
            print('step size smaller than threshold, exiting optimization')
            break
        """
    ####NOW WE GRADUALLY BRING THE MAGNITUDES OF THE MODS DOWN TO ZERO####
    
    minalphatol = 1e-10
    olddualval = dualval
    reductFactor = 1e-1
    reductCount = 1
    
    while True: #gradual reduction of modified source amplitude, outer loop
        fakeSratio *= reductFactor #if any new fake sources are added make them smaller too
        for i in range(len(fSlist)):
            fSlist[i] *= reductFactor

        alphaopt_grad = max(alphaopt_grad, 0.1)
        alphaopt_Hess = max(alphaopt_Hess, 0.1)
        
        iternum = 0
        while True:
            iternum += 1
            
            print('reducing fakeS now, at reduction #', reductCount, 'the iteration number is:', iternum, flush=True)
            
            doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
            
            dualval = dgHfunc(dof, dofgrad, dofHess, S1, fSlist, get_grad=True, get_Hess=(not doGD))
            objval = dualval - dof @ dofgrad
            abs_cstrt_sum = np.abs(dof) @ np.abs(dofgrad)
            print('current dual, objective, absolute sum of constraint violation are', dualval, objval, abs_cstrt_sum)
            
            if np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)): #objective convergence termination
                break
            
            if iternum % check_iter_period == 0:
                print('previous dual is', prevD)
                if np.abs(prevD-dualval)<np.abs(dualval)*1e-3: #dual convergence / stuck optimization termination
                    break
                if alphaopt_grad<minalphatol and alphaopt_Hess<minalphatol:
                    alphaopt_grad = 5e-5; alphaopt_Hess = 5e-5 #periodically boost the max step size since we are gradually turning off the modified sources
                prevD = dualval
            
            normgrad = np.linalg.norm(dofgrad)
            if not doGD:
                Ndir = np.linalg.solve(dofHess, -dofgrad)
                normNdir = np.linalg.norm(Ndir)
                pdir = Ndir / normNdir
                print('do regular Hessian step')
                print('normNdir is', normNdir)
                print('normgrad is', normgrad)
                print('Ndir dot grad is', np.dot(Ndir, dofgrad))
            if doGD:
                print('do regular gradient step')
                pdir = -dofgrad/normgrad
                print('normgrad is', normgrad)
                
            c1 = 0.5; c2 = 0.7 #the parameters for doing line search
            if doGD:
                alpha_start = alphaopt_grad
            else:
                alpha_start = alphaopt_Hess
            alpha = alpha_start
            
            print('alpha before feasibility backtrack', alpha)
            while validityfunc(dof+alpha*pdir)<=0:
                alpha *= c2
            
            alpha_feas = alpha
            print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = np.inf
            while True:
                tmp = dualfunc(dof+alpha*pdir)
                if tmp<Dopt: #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp; alphaopt=alpha
                else:
                    alphaopt=alpha
                    break
                if tmp<=dualval + c1*alpha*(pdir @ dofgrad): #Armijo backtracking condition
                    alphaopt = alpha
                    break
                alpha *= c2
            
            if alphaopt/alpha_start>(c2+1)/2: #in this case can start with bigger step
                alpha_newstart = alphaopt*2
            else:
                alpha_newstart = alphaopt
                if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                    mineigw, mineigv = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                
                    fakeSval = dgHfunc(dof, [], [], np.zeros_like(S1), [mineigv], get_grad=False, get_Hess=False)

                    epsS = np.sqrt(fakeSratio*np.abs(dualval/fakeSval))
                    fSlist.append(epsS * mineigv) #add new fakeS to fSlist
                    print('length of fSlist', len(fSlist))
            
            print('stepsize alphaopt is', alphaopt, '\n')
            dof += alpha*pdir
            if doGD:
                alphaopt_grad = alpha_newstart
            else:
                alphaopt_Hess = alpha_newstart
            """
            if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
                print('step size smaller than threshold, exiting optimization')
                break
            """
        """
        if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
            print('step size smaller than threshold, exiting reduce modS outer loop')
            break
        """
        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break #for now testing to see if we should just do away with lastReduct

        olddualval = dualval
        reductCount += 1
    
    print('time elapsed:', time.time()-tic, flush=True)
    return dof, dofgrad, dualval, objval
