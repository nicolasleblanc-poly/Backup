#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 16:57:05 2020

@author: pengning
"""

import numpy as np
import time

def get_spatialProj_dualgradHess_modS(dof, dofgrad, dofHess, dgHfunc, S1list, modSlist, epsSlist, get_grad=True, get_Hess=True):
    modenum = len(S1list)
    modS1list = []
    for mode in range(modenum):
        if epsSlist[mode]>0:
            modS1list.append(S1list[mode] + epsSlist[mode]*modSlist[mode])
        else:
            modS1list.append(S1list[mode])
    
    return dgHfunc(dof, dofgrad, dofHess, modS1list, get_grad=get_grad, get_Hess=get_Hess)
    

def modS_opt(initdof, S1list, dgHfunc, mineigfunc, opttol=1e-2, modSratio=1e-2, check_iter_period=20):
    """
    does modified source dual optimization for problems with a modal basis
    dgHfunc(dof, dofgrad, dofHess, S1list) returns the value of the dual given dof, and stores the gradient and Hessian in dofgrad and dofHess
    mineigfunc returns a tuple containing the mode # of the minimum eigenvalue, the minimum eigenvalue and the corresponding eigenvector confined within the modal family
    """
    modenum = len(S1list)
    
    modSlist = []
    epsSlist = [0]*modenum
    test_modSlist = []
    test_S1list = []
    test_epsSlist = [0]*modenum #these test_ lists are for helping to evaluate how big to set future modS
    for mode in range(modenum):
        modSlist.append(np.zeros_like(S1list[mode]))
        test_modSlist.append(np.zeros_like(S1list[mode]))
        test_S1list.append(np.zeros_like(S1list[mode]))
    
    dofnum = len(initdof)
    dof = initdof.copy() #.copy() because we don't wont to modify the initialization array in place
    dofgrad = np.zeros(dofnum)
    dofHess = np.zeros((dofnum,dofnum))
    
    tic = time.time()
    
    iternum = 0
    prevD = np.inf
    alphaopt_grad=1.0
    alphaopt_Hess=1.0
    tol_orthoS = 1e-5
    dualfunc = lambda d: get_spatialProj_dualgradHess_modS(d, np.array([]), np.array([]), dgHfunc, S1list, modSlist, epsSlist, get_grad=False, get_Hess=False)
    
    while True:
        iternum += 1
        
        print('the iteration number is:', iternum, flush=True)
        
        doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
        
        dualval = get_spatialProj_dualgradHess_modS(dof, dofgrad, dofHess, dgHfunc, S1list, modSlist, epsSlist, get_grad=True, get_Hess=(not doGD))
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
        while mineigfunc(dof+alpha*pdir)[1]<=0:
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
                singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                singular_mode, singular_eigw, singular_eigv = mineigfunc(singular_dof, eigvals_only=False)
                if epsSlist[singular_mode]<=0:
                    print('new modS aded at mode', singular_mode)
                    test_modSlist[singular_mode] = singular_eigv
                    test_epsSlist[singular_mode] = 1.0
                    modval = np.abs(get_spatialProj_dualgradHess_modS(singular_dof, np.array([]), np.array([]), dgHfunc, test_S1list, test_modSlist, test_epsSlist, get_grad=False, get_Hess=False))
                    modSlist[singular_mode] = singular_eigv
                    epsSlist[singular_mode] = np.sqrt(modSratio*np.abs(dualval/modval))
                    test_modSlist[singular_mode] = np.zeros_like(S1list[singular_mode])
                    test_epsSlist[singular_mode] = 0
                elif np.abs(np.vdot(singular_eigv, modSlist[singular_mode]))<tol_orthoS:
                    print('changed modS at mode', singular_mode)
                    modSlist[singular_mode] = np.sqrt(0.5)*(modSlist[singular_mode]+singular_eigv)
        
        print('stepsize alphaopt is', alphaopt, '\n')
        dof += alpha*pdir
        if doGD:
            alphaopt_grad = alpha_newstart
        else:
            alphaopt_Hess = alpha_newstart
        
    ####NOW WE GRADUALLY BRING THE MAGNITUDES OF THE MODS DOWN TO ZERO####
    alphaopt_grad = max(alphaopt_grad, 5e-5)
    alphaopt_Hess = max(alphaopt_Hess, 5e-5)
    minalphatol = 1e-10
    olddualval = dualval
    reductFactor = 1e-1
    reductCount = 1
    lastReduct = False
    
    while True: #gradual reduction of modified source amplitude, outer loop
        if not lastReduct:
            for i in range(len(epsSlist)):
                epsSlist[i] *= reductFactor
            modSratio *= reductFactor
        else:
            for i in range(len(epsSlist)):
                epsSlist[i] = 0
        
        iternum = 0
        while True:
            iternum += 1
            
            print('reducing modS now, at reduction #', reductCount, 'the iteration number is:', iternum, flush=True)
            
            doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
            
            dualval = get_spatialProj_dualgradHess_modS(dof, dofgrad, dofHess, dgHfunc, S1list, modSlist, epsSlist, get_grad=True, get_Hess=(not doGD))
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
            while mineigfunc(dof+alpha*pdir)[1]<=0:
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
                if (not lastReduct) and alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #don't bother to modify sources if this is the final reduction iteration
                    singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                    singular_mode, singular_eigw, singular_eigv = mineigfunc(singular_dof, eigvals_only=False)
                    if epsSlist[singular_mode]<=0:
                        print('new modS aded at mode', singular_mode)
                        test_modSlist[singular_mode] = singular_eigv
                        test_epsSlist[singular_mode] = 1.0
                        modval = np.abs(get_spatialProj_dualgradHess_modS(singular_dof, np.array([]), np.array([]), dgHfunc, test_S1list, test_modSlist, test_epsSlist, get_grad=False, get_Hess=False))
                        modSlist[singular_mode] = singular_eigv
                        epsSlist[singular_mode] = np.sqrt(modSratio*np.abs(dualval/modval))
                        test_modSlist[singular_mode] = np.zeros_like(S1list[singular_mode])
                        test_epsSlist[singular_mode] = 0
                    elif np.abs(np.vdot(singular_eigv, modSlist[singular_mode]))<tol_orthoS:
                        print('changed modS at mode', singular_mode)
                        modSlist[singular_mode] = np.sqrt(0.5)*(modSlist[singular_mode]+singular_eigv)
            
            print('stepsize alphaopt is', alphaopt, '\n')
            dof += alpha*pdir
            if doGD:
                alphaopt_grad = alpha_newstart
            else:
                alphaopt_Hess = alpha_newstart
        
        if lastReduct:
            break
        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break #for now testing to see if we should just do away with lastReduct
            lastReduct = True
        olddualval = dualval
        reductCount += 1
    
    print('time elapsed:', time.time()-tic, flush=True)
    return dof, dofgrad, dualval, objval