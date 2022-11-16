#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:57:05 2020

@author: pengning
"""

import numpy as np
import time

def get_spatialProj_dualgradHess_modS_singlematrix(dof, dofgrad, dofHess, dgHfunc, S1, modS, epsS, get_grad=True, get_Hess=True):
    totS1 = S1 + epsS * modS
    return dgHfunc(dof, dofgrad, dofHess, totS1, get_grad=get_grad, get_Hess=get_Hess)
    

def modS_opt_singlematrix(initdof, S1, dgHfunc, validityfunc, modSfunc, opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    """
    does modified source dual optimization
    dgHfunc(dof, dofgrad, dofHess, S1) returns the value of the dual given dof, and stores the gradient and Hessian in dofgrad and dofHess
    validityfunc(dof) returns 1 if dof is within domain of duality and -1 otherwise
    modSfunc(dof) returns a normalized vector for use as modS
    """
    
    modS = np.zeros_like(S1, dtype=np.complex)
    epsS = -1.0
    test_modS = np.zeros_like(S1, dtype=np.complex)
    test_S1 = np.zeros_like(S1, dtype=np.complex)
    test_epsS = 0.0 #this is for helping evaluate how big to set future modS
    ZTTeigv = None #used to check if we need to update modS
    
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
    dualfunc = lambda d: get_spatialProj_dualgradHess_modS_singlematrix(d, [], [], dgHfunc, S1, modS, epsS, get_grad=False, get_Hess=False)

    while True:
        iternum += 1
        
        print('the iteration number is:', iternum, flush=True)
        
        doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
        
        dualval = get_spatialProj_dualgradHess_modS_singlematrix(dof, dofgrad, dofHess, dgHfunc, S1, modS, epsS, get_grad=True, get_Hess=(not doGD))
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
            if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add mod source
                print('encountered feasibility wall')
                singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                test_modS, test_ZTTeigv = modSfunc(singular_dof)
                
                if epsS<=0:
                    print('modS added')
                    ZTTeigv = test_ZTTeigv.copy()
                    test_epsS = 1.0
                    modval = np.abs(get_spatialProj_dualgradHess_modS_singlematrix(singular_dof, [], [], dgHfunc, test_S1, test_modS, test_epsS, get_grad=False, get_Hess=False))
                    epsS = np.sqrt(modSratio*np.abs(dualval/modval))
                    modS = test_modS.copy()
                    print('epsS takes new value', epsS)
                    test_modS = np.zeros_like(S1)
                    test_epsS = 0
                elif np.abs(np.vdot(test_ZTTeigv, ZTTeigv))<tol_orthoS:
                    print('changed modS')
                    ZTTeigv = test_ZTTeigv.copy()
                    modS = np.sqrt(0.5)*(modS+test_modS)
        
        print('stepsize alphaopt is', alphaopt, '\n')
        dof += alpha*pdir
        if doGD:
            alphaopt_grad = alpha_newstart
        else:
            alphaopt_Hess = alpha_newstart
        if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
            print('step size smaller than threshold, exiting optimization')
            break
        
    ####NOW WE GRADUALLY BRING THE MAGNITUDES OF THE MODS DOWN TO ZERO####
    alphaopt_grad = max(alphaopt_grad, 5e-5)
    alphaopt_Hess = max(alphaopt_Hess, 5e-5)
    minalphatol = 1e-10
    olddualval = dualval
    reductFactor = 1e-1
    reductCount = 1
    
    while True: #gradual reduction of modified source amplitude, outer loop
        epsS *= reductFactor
        modSratio *= reductFactor
        
        iternum = 0
        while True:
            iternum += 1
            
            print('reducing modS now, at reduction #', reductCount, 'the iteration number is:', iternum, flush=True)
            
            doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
            
            dualval = get_spatialProj_dualgradHess_modS_singlematrix(dof, dofgrad, dofHess, dgHfunc, S1, modS, epsS, get_grad=True, get_Hess=(not doGD))
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
                    ##SHOULD MAKE THIS MORE TRANSPARENT IN THE FUTURE##
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
                if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #don't bother to modify sources if this is the final reduction iteration
                    singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                    test_modS, test_ZTTeigv = modSfunc(singular_dof)
                    #test_modS /= np.linalg.norm(test_modS)
                    
                    if epsS<=0:
                        print('modS added')
                        ZTTeigv = test_ZTTeigv.copy()
                        test_epsS = 1.0
                        modval = np.abs(get_spatialProj_dualgradHess_modS_singlematrix(singular_dof, [], [], dgHfunc, test_S1, test_modS, test_epsS, get_grad=False, get_Hess=False))
                        epsS = np.sqrt(modSratio*np.abs(dualval/modval))
                        modS = test_modS.copy()
                        test_modS = np.zeros_like(S1)
                        test_epsS = 0
                    elif np.abs(np.vdot(test_ZTTeigv, ZTTeigv))<tol_orthoS:
                        print('changed modS')
                        ZTTeigv = test_ZTTeigv.copy()
                        modS = np.sqrt(0.5)*(modS+test_modS)
            
            print('stepsize alphaopt is', alphaopt, '\n')
            dof += alpha*pdir
            if doGD:
                alphaopt_grad = alpha_newstart
            else:
                alphaopt_Hess = alpha_newstart
            if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
                print('step size smaller than threshold, exiting optimization')
                break
        if alphaopt_grad<1e-10 and alphaopt_Hess<1e-10:
            print('step size smaller than threshold, exiting reduce modS outer loop')
            break
        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break #for now testing to see if we should just do away with lastReduct

        olddualval = dualval
        reductCount += 1
    
    print('time elapsed:', time.time()-tic, flush=True)
    return dof, dofgrad, dualval, objval
