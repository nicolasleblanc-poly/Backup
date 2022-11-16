#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:09:01 2021

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc, gradConverge=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    
    dofnum = len(initdof)
    grad = np.zeros(dofnum)
    tmp_grad = np.zeros(dofnum)
    Hinv = np.eye(dofnum) #approximate inverse Hessian
    
    prev_dualval = np.inf
    dof = initdof
    
    justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)

    olddualval = np.inf
    reductCount = 0
    
    while True: #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration

        alpha_last = 1.0 #reset step size
        Hinv = np.eye(dofnum) #reset Hinv
        
        dualval = dgfunc(dof, grad, fSlist, get_grad=True)
        print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
        while True:
            iternum += 1
            print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)
            
            Ndir = - Hinv @ grad
            pdir = Ndir / la.norm(Ndir)
        
            #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = pdir @ grad
            print('pdir dot grad is', p_dot_grad)
            c_reduct = 0.7; c_A = 1e-4; c_W = 0.9
            alpha = alpha_start = alpha_last
            print('starting alpha', alpha_start)
            while validityfunc(dof+alpha*pdir)<=0: #move back into feasibility region
                alpha *= c_reduct
            alpha_feas = alpha
            print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = np.inf
            while True:
                tmp_dual = justfunc(dof+alpha*pdir)
                if tmp_dual<Dopt: #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp_dual; alphaopt=alpha
                else:
                    alphaopt=alpha ###ISSUE!!!!
                    break
                if tmp_dual<=dualval + c_A*alpha*p_dot_grad: #Armijo backtracking condition
                    alphaopt = alpha
                    break
                alpha *= c_reduct
        
            added_fakeS = False
            if alphaopt/alpha_start>(c_reduct+1)/2: #in this case can start with bigger step
                alpha_last = alphaopt*2
            else:
                alpha_last = alphaopt
                if alpha_feas/alpha_start<(c_reduct+1)/2 and alphaopt/alpha_feas>(c_reduct+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = True
                    print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + alpha_feas*pdir #dof that is roughly on duality boundary
                    mineigw, mineigv = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                    
                    fakeSval = dgfunc(dof, np.array([]), [mineigv], get_grad=False)
                    
                    epsS = np.sqrt(fakeSratio*np.abs(dualval/fakeSval))
                    fSlist.append(epsS * mineigv) #add new fakeS to fSlist
                    print('length of fSlist', len(fSlist))

            print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir

            #########decide how to update Hinv############
            tmp_dual = dgfunc(dof+delta, tmp_grad, fSlist, get_grad=True)
            p_dot_tmp_grad = pdir @ tmp_grad
            if added_fakeS:
                Hinv = np.eye(dofnum) #the objective has been modified; restart Hinv from identity
            elif p_dot_tmp_grad > c_W*p_dot_grad: #satisfy Wolfe condition, update Hinv
                print('updating Hinv')
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv@tmp_grad + Ndir
                Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) 
                          - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
        
        
            dualval = tmp_dual
            grad[:] = tmp_grad[:]
            dof += delta
        
            objval = dualval - np.dot(dof,grad)
            eqcstval = np.abs(dof) @ np.abs(grad)
            print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
            print('normgrad is', la.norm(grad))
              
            if gradConverge and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval) and la.norm(grad)<opttol * np.abs(dualval): #objective and gradient norm convergence termination
                break

            if (not gradConverge) and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval): #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
        
            if iternum % iter_period==0:
                print('prev_dualval is', prev_dualval)
                if np.abs(prev_dualval-dualval)<np.abs(dualval)*opttol: #dual convergence / stuck optimization termination
                    break
                prev_dualval = dualval

        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break
        """
        #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
            #break
        if len(fSlist)==0: #converged without needing fake sources, suggests strong duality
            break
        """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    
    return dof, grad, dualval, objval
    
