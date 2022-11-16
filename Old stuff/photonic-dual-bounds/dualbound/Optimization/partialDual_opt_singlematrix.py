#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 16:49:09 2021

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import time

def partialDual_opt_singlematrix(initdof, pDfunc, opttol=1e-2, iterperiod=20):
    
    dofnum = len(initdof)
    dof = initdof.copy()
    dofgrad = np.zeros(dofnum)
    dofHess = np.zeros((dofnum,dofnum))
    
    iternum = 0
    prevD = np.inf
    alphaopt_grad=1.0
    alphaopt_Hess=1.0
    
    dualfunc = lambda d: pDfunc(d, np.array([]), np.array([]))[0]
    
    tic = time.time()
    
    while True: #optimization
        iternum += 1

        print('the iteration number is:', iternum, flush=True)
        
        doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a 
        
        if not doGD:
            dualval, pDData = pDfunc(dof, dofgrad, dofHess)
        else:
            dualval, pDdata = pDfunc(dof, dofgrad, np.array([]))
        

        objval = dualval - np.dot(dof,dofgrad)
        eqcstval = np.abs(dof) @ np.abs(dofgrad)
        print('current dual, objective, eqconstraint values are', dualval, objval, eqcstval)

        if np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval) and la.norm(dofgrad)<opttol*np.abs(objval): #obj. convergence termination
            break
        
        if iternum % iterperiod==0:
            print('prevD is', prevD)
            if np.abs(prevD-dualval)<np.abs(dualval)*opttol: #dual convergence / stuck optimization termination
                break
            
            prevD = dualval
        
        if not pDData['flagRegular']:
            doGD = True #for a special solution we only calculate its gradient
            
        normgrad = la.norm(dofgrad)

        if not doGD:
            Ndir = la.solve(dofHess, -dofgrad)
            normNdir = la.norm(Ndir)
            pdir = Ndir / normNdir
            print('do regular Hessian step')
            print('normNdir is', normNdir)
            print('normgrad is', normgrad)
            print('Ndir dot grad is', np.dot(Ndir, dofgrad))
            
        if doGD:
            print('do regular gradient step')
            pdir = -dofgrad/normgrad
            print('normgrad is', normgrad)
            
        if True: #for now it seems not doing exact line search but just Armijo is better
            if doGD:
                alpha = alphaopt_grad
            else:
                alpha = alphaopt_Hess
            
            print('alpha before backtracking is', alpha)
            alphaopt = alpha; alphastart = alpha
            Dopt = np.inf
            while True:
                tmp = dualfunc(dof+alpha*pdir)
                if tmp<Dopt:
                    Dopt = tmp; alphaopt=alpha
                else:
                    alphaopt=alpha
                    break
                if tmp<=dualval+0.5*alpha*np.dot(pdir, dofgrad):
                    break
                alpha *= 0.7
            #alphaopt=alpha
            if alphaopt/alphastart>0.8: #in this case can start with bigger step
                alphastart = alphaopt*2
            else:
                alphastart = alphaopt
        print('stepsize alphaopt is', alphaopt, '\n')
        dof += pdir*alphaopt
        if doGD:
            alphaopt_grad = alphastart
        else:
            alphaopt_Hess = alphastart
        
    print('time elapsed:', time.time()-tic, flush=True)
    return dof, dofgrad, dualval, objval
