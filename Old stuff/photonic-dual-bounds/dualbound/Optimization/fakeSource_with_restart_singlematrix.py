import numpy as np
import scipy.linalg as la
import time
    

def fakeS_with_restart_singlematrix(initdof, dgHfunc, validityfunc, mineigfunc, opttol=1e-2, gradConverge=False, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    """
    does fake source dual optimization
    dgHfunc(dof, dofgrad, dofHess, fSlist) returns the value of the dual+\sum <fS_i|ZTT^-1|fS_i> given dof, and stores the gradient and Hessian in dofgrad and dofHess
    validityfunc(dof) returns 1 if dof is within domain of duality and -1 otherwise
    fSlist starts off empty and the algorithm adds in fake source terms as necessary
    nested iteration structure; at end of inner iteration remove all fakeSources and start over, with smaller fakeSratio
    """

    fSlist = []
    
    dofnum = len(initdof)
    dof = initdof.copy() #.copy() because we don't wont to modify the initialization array in place
    dofgrad = np.zeros(dofnum)
    dofHess = np.zeros((dofnum,dofnum))
    
    tic = time.time()
    
    dualfunc = lambda d: dgHfunc(d, [],[], fSlist, get_grad=False, get_Hess=False)

    olddualval = np.inf
    reductCount = 0

    while True: #outer loop; gradually reduce amplitudes of fake sources
        fSlist = [] #remove old fake sources and restart with each outer iteration

        alphaopt_grad=1.0
        alphaopt_Hess=1.0 #reset step size

        iternum = 0
        prevD = np.inf

        print('\n At Outer iteration loop #', reductCount, '\n', flush=True)
        while True:
            iternum += 1
        
            print('Outer iteration #', reductCount, 'the iteration number is:', iternum, flush=True)
        
            doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
        
            dualval = dgHfunc(dof, dofgrad, dofHess, fSlist, get_grad=True, get_Hess=(not doGD))
            normgrad = la.norm(dofgrad)
            print('normgrad is', normgrad)
            objval = dualval - dof @ dofgrad
            abs_cstrt_sum = np.abs(dof) @ np.abs(dofgrad)
            print('current dual, objective, absolute sum of constraint violation are', dualval, objval, abs_cstrt_sum)

            #modify condition to try and minimize gradient as well?
            if gradConverge and iternum>min_iter and np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)) and normgrad<opttol*min(np.abs(objval),np.abs(dualval)): #objective and gradient norm convergence termination
                break

            if (not gradConverge) and iternum>min_iter and np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)): #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            
            if iternum % iter_period == 0:
                print('previous dual is', prevD)
                if np.abs(prevD-dualval)<np.abs(dualval)*1e-3: #dual convergence / stuck optimization termination
                    print('stuck optimization, exiting inner loop')
                    break
                prevD = dualval
        
            if not doGD:
                Ndir = la.solve(dofHess, -dofgrad)
                normNdir = la.norm(Ndir)
                pdir = Ndir / normNdir
                print('do regular Hessian step')
                print('normNdir is', normNdir)
                print('pdir dot grad is', np.dot(pdir, dofgrad))
            if doGD:
                print('do regular gradient step')
                pdir = -dofgrad/normgrad
            
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
                    break
                #did away with Armijo condition b.c. we know that dual is convex
                alpha *= c2
        
            if alphaopt/alpha_start>(c2+1)/2: #in this case can start with bigger step
                alpha_newstart = alphaopt*2
            else:
                alpha_newstart = alphaopt

            if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                print('encountered feasibility wall, adding a fake source term')
                singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                mineigw, mineigv = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                
                fakeSval = dgHfunc(dof, [], [], fSlist+[mineigv], get_grad=False, get_Hess=False)

                epsS = np.sqrt(fakeSratio*np.abs(dualval/(fakeSval-dualval)))
                fSlist.append(epsS * mineigv) #add new fakeS to fSlist
                print('length of fSlist', len(fSlist))
        
            print('stepsize alphaopt is', alphaopt, '\n')
            dof += alphaopt*pdir
            if doGD:
                alphaopt_grad = alpha_newstart
            else:
                alphaopt_Hess = alpha_newstart

        ###################BACK TO OUTER LOOP##################
        if np.abs(olddualval-dualval)<opttol*np.abs(dualval):
            break

        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration

    ########END OF OUTER LOOP###############
    return dof, dofgrad, dualval, objval



def fakeS_single_restart_Newton(initdof, dgHfunc, validityfunc, mineigfunc, fSlist=[], opttol=1e-2, gradConverge=True, singlefS=True, fakeSratio=1e-2, iter_period=20, min_iter=6):
    """
    does fake source dual optimization with a single restart, for use with dual space reduction iteration
    dgHfunc(dof, dofgrad, dofHess, fSlist) returns the value of the dual+\sum <fS_i|ZTT^-1|fS_i> given dof, and stores the gradient and Hessian in dofgrad and dofHess
    validityfunc(dof) returns 1 if dof is within domain of duality and -1 otherwise
    fSlist can be NON-empty at start and the algorithm adds in fake source terms as necessary
    nested iteration structure; at end of inner iteration remove all fakeSources and start over once to generate new fSlist given updated constraints
    """

    print('length of fSlist at the beginning', len(fSlist)) #check mutable object as default parameter behavior
    
    dofnum = len(initdof)
    dof = initdof.copy() #.copy() because we don't wont to modify the initialization array in place
    dofgrad = np.zeros(dofnum)
    dofHess = np.zeros((dofnum,dofnum))
    
    tic = time.time()
    
    dualfunc = lambda d: dgHfunc(d, [],[], fSlist, get_grad=False, get_Hess=False)

    reductCount = 0 #outer iteration number

    while True: #outer loop; gradually reduce amplitudes of fake sources

        alphaopt_grad=1.0
        alphaopt_Hess=1.0 #reset step size

        iternum = 0
        prevD = np.inf

        print('\n At Outer iteration loop #', reductCount, '\n', flush=True)
        while True:
            iternum += 1
        
            print('Outer iteration #', reductCount, 'the iteration number is:', iternum, flush=True)
        
            doGD = (iternum % 2 == 0) #flag for deciding whether to do a gradient step or a Newton step
        
            dualval = dgHfunc(dof, dofgrad, dofHess, fSlist, get_grad=True, get_Hess=(not doGD))
            normgrad = la.norm(dofgrad)
            print('normgrad is', normgrad)
            objval = dualval - dof @ dofgrad
            abs_cstrt_sum = np.abs(dof) @ np.abs(dofgrad)

            print('current dual+fakeS, objective+fakeS, absolute sum of constraint violation are', dualval, objval, abs_cstrt_sum)
            #check mineig of ZTT to see how fake sources are doing in this respect
            mineigw, mineigv = mineigfunc(dof)
            print('mineig of ZTT is', mineigw)

            #modify condition to try and minimize gradient as well?
            if gradConverge and iternum>min_iter and np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)) and normgrad<opttol*min(np.abs(objval),np.abs(dualval)): #objective and gradient norm convergence termination
                break

            if (not gradConverge) and iternum>min_iter and np.abs(dualval-objval)<opttol*min(np.abs(objval),np.abs(dualval)) and abs_cstrt_sum<opttol*min(np.abs(objval),np.abs(dualval)): #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            
            if iternum % iter_period == 0:
                print('previous dual is', prevD)
                if np.abs(prevD-dualval)<np.abs(dualval)*1e-3: #dual convergence / stuck optimization termination
                    print('stuck optimization, exiting inner loop')
                    break
                prevD = dualval
        
            if not doGD:
                Ndir = la.solve(dofHess, -dofgrad)
                normNdir = la.norm(Ndir)
                pdir = Ndir / normNdir
                print('do regular Hessian step')
                print('normNdir is', normNdir)
                print('pdir dot grad is', np.dot(pdir, dofgrad))
            if doGD:
                print('do regular gradient step')
                pdir = -dofgrad/normgrad
            
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
                    break
                #did away with Armijo condition b.c. we know that dual is convex
                alpha *= c2
        
            if alphaopt/alpha_start>(c2+1)/2: #in this case can start with bigger step
                alpha_newstart = alphaopt*2
            else:
                alpha_newstart = alphaopt

            if alpha_feas/alpha_start<(c2+1)/2 and alphaopt/alpha_feas>(c2+1)/2: #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                print('encountered feasibility wall, adding a fake source term')
                #singular_dof = dof + (alpha_feas/c2)*pdir #dof that is roughly on duality boundary
                #singular_dof = dof
                singular_dof = dof + alphaopt*pdir
                print('check PD of singular_dof', validityfunc(singular_dof))
                mineigw, mineigv = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                print('mineigw', mineigw)
                
                #fakeSval = dgHfunc(dof, [], [], fSlist+[mineigv], get_grad=False, get_Hess=False)
                #epsS = np.sqrt(fakeSratio*np.abs(dualval/(fakeSval-dualval)))

                justdual = dgHfunc(dof, [], [], [], get_grad=False, get_Hess=False)
                fakeSval = dgHfunc(dof, [], [], [mineigv], get_grad=False, get_Hess=False) - justdual
                epsS = np.sqrt(fakeSratio * np.abs(justdual / fakeSval))
                
                fSlist.append(epsS * mineigv) #add new fakeS to fSlist
                print('length of fSlist', len(fSlist))
        
            print('stepsize alphaopt is', alphaopt, '\n')
            dof += alphaopt*pdir
            if doGD:
                alphaopt_grad = alpha_newstart
            else:
                alphaopt_Hess = alpha_newstart
            
        ###################BACK TO OUTER LOOP##################
        #reductCount>0 so we update fSlist once every optimization iteration
        if reductCount>0:
            break
        fSlist = [] #remove old fake sources and restart with each outer iteration
        reductCount += 1

    ########END OF OUTER LOOP###############
    return dof, dofgrad, dualval, objval, fSlist #return fSlist for constructing new projection constraint

