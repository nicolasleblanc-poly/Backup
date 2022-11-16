#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
implements BFGS-B for primal inequality constraints taking into account dual feasibility constraints
"""

import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt


def get_m(x, x0, g, B):
    delx = x-x0
    return np.dot(g,delx) + 0.5*np.dot(delx, B @ delx)

def get_max_stepsize(x0, l, u, pdir):
    if len(x0)<1:
        return np.inf
    alphalist = np.inf * np.ones_like(x0)
    alphalist[pdir>0] = (u[pdir>0]-x0[pdir>0]) / pdir[pdir>0]
    alphalist[pdir<0] = (l[pdir<0]-x0[pdir<0]) / pdir[pdir<0]
    return np.min(alphalist)

def get_descent_dir(x0, g, Hinv, l, u):
    """
    following Byrd, Lu, Nocedal, Zhu (1994)
    use a gradient projection method to find decent direction
    """
    tlist = np.zeros_like(g)
    tlist[g<0] = (x0[g<0]-u[g<0])/g[g<0]
    tlist[g>0] = (x0[g>0]-l[g>0])/g[g>0]
    tlist[g==0] = np.inf

    sorted_tind =np.argsort(tlist)
    sorted_tlist = tlist[sorted_tind]

    B = la.inv(Hinv)
    print('Hessian approx cond #', la.norm(B,2)*la.norm(Hinv,2))
    d0 = np.zeros_like(g)
    d0[0<tlist] = -g[0<tlist]
    f1p = np.dot(g, d0)
    f2p = np.dot(d0, B @ d0)

    zj = np.zeros_like(x0)
    dprev = d0
    tprev = 0
    topt = sorted_tlist[-1] #if no minimizer found after sweeping through all t intervals this will become step size
    for j in range(len(x0)): #supposedly significant speed-ups using LBFGS-B, investigate
        tj = sorted_tlist[j]
        if not tj>0:
            continue #skip over all the already active constraints, which have already been zeroed out in d0/dprev
        
        deltat_opt = -f1p / f2p
        if deltat_opt>=0 and tprev+deltat_opt<tj:
            topt = tprev+deltat_opt
            break
        elif f1p>=0:
            topt = tprev
            break

        zj += (tj-tprev) * dprev
        gb = g[sorted_tind[j]]
        B_eb = B[:,sorted_tind[j]]
        f1p += gb**2 + (tj-tprev)*f2p + gb*np.dot(B_eb, zj)
        f2p += 2*gb*np.dot(B_eb, dprev) + gb**2*B_eb[sorted_tind[j]]

        tprev = tj
        dprev[sorted_tind[j]] = 0.0

    xc = x0 - topt*g
    active = np.logical_or(xc>=u,xc<=l)
    xc[xc>u] = u[xc>u]
    xc[xc<l] = l[xc<l]

    #next optimize inactive dofs around gradient projected point xc
    xopt = xc.copy()
    if int(np.sum(active))==len(x0):
        return xopt

    #print('active', active)
    inactive = np.logical_not(active)
    B_red = B[np.ix_(inactive,inactive)]
    r_red = (g + B@(xc-x0))[inactive]
    d_opt = -la.solve(B_red,r_red)
    
    x_inactive = xc[inactive]
    l_inactive = l[inactive]
    u_inactive = u[inactive]
    alphalist = np.ones_like(x_inactive) * np.inf
    alphalist[d_opt>0] = (u_inactive[d_opt>0]-x_inactive[d_opt>0]) / d_opt[d_opt>0]
    alphalist[d_opt<0] = (l_inactive[d_opt<0]-x_inactive[d_opt<0]) / d_opt[d_opt<0]

    full_alphalist = np.zeros_like(x0)
    full_alphalist[inactive] = alphalist
    print('d-alpha full_alphalist', full_alphalist)
    
    alpha = min(1.0, np.min(alphalist))
    print('d-alpha val', alpha)
    xopt[inactive] += alpha*d_opt
    xopt[xopt<l] = l[xopt<l]
    xopt[xopt>u] = u[xopt>u] #numerical safeguard
    print('x0', x0)
    print('xc', xc)
    print('normdiff of xc and x0', la.norm(xc-x0))
    print('active', active)
    print('inactive', inactive)
    print('xopt', xopt)
    
    Ndir = xopt-x0
    return Ndir



def BFGS_with_ineq(initdof, is_ineq, dgfunc, validityfunc, opttol=1e-2, iter_period=20, verbosefunc=None):
    """
    bare BFGS with no fake sources; used for testing the optimization landscape
    """
    dofnum = len(initdof)
    grad = np.zeros(dofnum)
    tmp_grad = np.zeros(dofnum)
    Hinv = np.eye(dofnum) #approximate inverse Hessian

    lbound = -np.inf * np.ones_like(initdof)
    lbound[is_ineq] = 0.0
    rbound = np.inf * np.ones_like(initdof)
    
    iternum = 0
    prev_dualval = np.inf
    alpha_last = 1.0 #some memory of the step size
    dof = initdof
    
    fSlist = [] #no fake sources
    
    dualval = dgfunc(dof, grad, fSlist, get_grad=True)
    justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    
    print('at initdof the dual value is', dualval)
    
    while True:
        iternum += 1
        print('the iteration number is:', iternum, flush=True)

        Ndir = get_descent_dir(dof, grad, Hinv, lbound, rbound)
        Ndirnorm = la.norm(Ndir)
        pdir = Ndir/Ndirnorm
        alpha_max = get_max_stepsize(dof[is_ineq], lbound[is_ineq], rbound[is_ineq], pdir[is_ineq])
        
        #backtracking line search, impose feasibility and Armijo condition
        p_dot_grad = pdir @ grad
        print('normgrad is', la.norm(grad), 'pdir dot grad is', p_dot_grad)
        c_reduct = 0.7; c_A = 1e-4; c_W = 0.9
        alpha = min(alpha_last, alpha_max) #so we never violate bound constraints
        print('alpha_last', alpha_last, 'alpha_max', alpha_max, 'starting alpha', alpha)
        while validityfunc(dof+alpha*pdir)<=0: #move back into feasibility region
            alpha *= c_reduct
        alpha_feas = alpha
        print('alpha before backtracking is', alpha_feas)
        alpha_opt = alpha
        Dopt = np.inf
        while True:
            tmp_dual = justfunc(dof+alpha*pdir)
            if tmp_dual<Dopt: #the dual is still decreasing as we backtrack, continue
                Dopt = tmp_dual; alpha_opt=alpha
            else:
                alpha_opt=alpha
                break
            if tmp_dual<=dualval + c_A*alpha*p_dot_grad: #Armijo backtracking condition
                alpha_opt = alpha
                break
            alpha *= c_reduct
        
        if alpha_opt/alpha_last>(c_reduct+1)/2: #in this case can start with bigger step
            alpha_last = alpha_opt*2
        else:
            alpha_last = alpha_opt
            
        print('stepsize alpha_opt is', alpha_opt, '\n')
        delta = alpha_opt * pdir

        #########decide how to update Hinv############
        tmp_dual = dgfunc(dof+delta, tmp_grad, fSlist, get_grad=True)
        p_dot_tmp_grad = pdir @ tmp_grad
        
        if p_dot_tmp_grad > c_W*p_dot_grad: #satisfy Wolfe condition, update Hinv
            print('updating Hinv')
            gamma = tmp_grad - grad
            gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
            Hinv_dot_gamma = Hinv@gamma
            Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) 
                      - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
        
        
        dualval = tmp_dual
        grad[:] = tmp_grad[:]
        dof += delta
        
        objval = dualval - np.dot(dof,grad)
        eqcstval = np.abs(dof) @ np.abs(grad)
        print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)

        violation = grad.copy()
        violation[np.logical_and(is_ineq,grad>=0)] = 0.0
        
        if verbosefunc is None:
            print('dof is', dof)
            print('normgrad is', la.norm(grad))
            print('grad is', grad)
            print('norm violation is', la.norm(violation))
        else:
            verbosefunc(dof, grad) #for flexibility in outputting, for research purposes
            
        if np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval) and la.norm(violation)<opttol * np.abs(dualval):
            break
        
        if iternum % iter_period==0:
            print('prev_dualval is', prev_dualval)
            if np.abs(prev_dualval-dualval)<np.abs(dualval)*opttol: #dual convergence / stuck optimization termination
                break
            prev_dualval = dualval

    
    return dof, grad, dualval, objval
    
