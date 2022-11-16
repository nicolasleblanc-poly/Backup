#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 25 15:09:01 2021

@author: pengning
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def BFGS(func, initdof, opttol=1e-2, iter_period=10, maxiter=None, stopval=None, ftol=None, plot_iters=False):
    if maxiter is None:
        maxiter=1000 #default maximum iteration #
    
    dofnum = len(initdof)
    grad = np.zeros(dofnum)
    tmp_grad = np.zeros(dofnum)
    Hinv = np.eye(dofnum) #approximate inverse Hessian
    
    iternum = 0
    prev_val = np.inf
    last_alpha = 1.0 #some memory of the step size
    dof = initdof
        
    val = func(dof, grad)
    if plot_iters:
        val_list = [val]
    print('at initdof the func value is', val)
    
    while True:
        iternum += 1
        print('the iteration number is:', iternum, flush=True)
            
        Ndir = - Hinv @ grad
        pdir = Ndir / la.norm(Ndir)
        
        #inexact line search, impose Armijo and weak Wolfe condition
        p_dot_grad = pdir @ grad
        print('pdir norm is', la.norm(pdir), 'p_dot_grad is', p_dot_grad)
        c1 = 1e-4; c2 = 0.9 #parameters suggested from Nocedal and Wright
        l_alpha=0; r_alpha=np.inf
        alpha = last_alpha
        search_success = True
        while True:
            print('l_alpha', l_alpha, 'r_alpha', r_alpha, 'alpha', alpha)
            delta = alpha*pdir
            tmp_val = func(dof + delta, tmp_grad)
            #print('estimate pdotgrad is', (tmp_val-val)/alpha)
            p_dot_tmp_grad = pdir @ tmp_grad
            #print('tmp_val', tmp_val, 'p_dot_tmp_grad', p_dot_tmp_grad)
            if tmp_val > val + c1*alpha*p_dot_grad: #fail Armijo condition
                print('failed Armijo')
                r_alpha = alpha
            elif p_dot_tmp_grad < c2*p_dot_grad: #fail weak Wolfe condition
                print('failed Wolfe')
                l_alpha = alpha
            else: #satisfy both conditions
                break
            if r_alpha < np.inf:
                alpha = (l_alpha + r_alpha)/2.0
                if (r_alpha-l_alpha)/r_alpha < 1e-8:
                    print('unable to find suitable step size')
                    search_success = False
                    break
            else:
                alpha *= 2

        if not search_success: #encountered numerical difficulties, stopping
            break

        print('found alpha', alpha)
        last_alpha = alpha

        if search_success:
            #BFGS update
            #delta = x^{k+1}-x^k     gamma = grad^{k+1}-grad^k
            gamma = tmp_grad - grad
            gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
            Hinv_dot_gamma = Hinv@tmp_grad + Ndir
            
            Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) 
                      - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
        
        
        val = tmp_val
        grad[:] = tmp_grad[:]
        dof += delta
        if plot_iters:
            val_list.append(val)
        
        print('at iteration #', iternum, 'the func value is', val)
        print('normgrad is', la.norm(grad))

        if (not stopval is None) and val<stopval:
            print('reached stopval termination condition')
            break
        if (not maxiter is None) and iternum>maxiter:
            print('max iteration number reached')
            break
        if (not ftol is None) and iternum % iter_period==0:
            print('prev_val is', prev_val)
            if np.abs(prev_val-val)<np.abs(val)*ftol: #dual convergence / stuck optimization termination
                break
            prev_val = val

        
    if plot_iters:
        return dof, grad, val, val_list
    else:
        return dof, grad, val
    
