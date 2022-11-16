import numpy as np
import math
import scipy.linalg as la
import subprocess
import os
from A_matrix import Am
from b_vector import bv
from T_Solver import Dual 

def mineigfunc(x, chi_invdag, Gdag, Pv):
    A=Am(x, chi_invdag, Gdag, Pv)
    # print(A)
    w_val,v=la.eig(A)
    # w: eigen values
    # v: eigen vectors
    w=np.real(w_val)
    # This code is to find the minimum eigenvalue
    min_eval=w[-1]
    # print(min_eval)
    # This code is to find the eigenvector
    # associated with the mimimum eigenvalue
    dim=A.shape[0]
    min_evec=v[:,-1].reshape(dim,1)
    return min_eval,min_evec
def validityfunc(x, chi_invdag, Gdag, Pv):
    val=mineigfunc(x, chi_invdag, Gdag, Pv)
    min_eval=val[0]
    min_evec=val[1]
    if min_eval>0:
        return 1
    else:
        return -1
def BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, e_vac, gradConverge=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    dofnum = len(initdof)
    grad = np.zeros(dofnum)
    tmp_grad = np.zeros(dofnum)
    Hinv = np.eye(dofnum) #approximate inverse Hessian
    prev_dualval = np.inf
    dof = initdof
    justfunc = lambda d: dgfunc(d, np.array([]), fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, False, e_vac)
    olddualval = np.inf
    reductCount = 0
    it=0
    while True: #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = np.eye(dofnum,dtype='complex') #reset Hinv
        val = dgfunc(dof, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, True, e_vac)  
        dualval = val[0]
        grad = val[1]
        obj = val[2]
        T = val[3]
        A=val[4]
        b=val[5]
        print("A.shape ",A.shape,"\n")
        print("b.shape ",b.shape,"\n")
    #             print("grad",grad,"\n")
                # This gets the value of the dual function and the dual's gradient
    #             print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
        while True:
            # print("Hinv",Hinv,"\n")
            iternum += 1
    #                 print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)
            Ndir = - Hinv @ grad
            pdir = Ndir / la.norm(Ndir)
                    #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = pdir @ grad
    #                 print('pdir dot grad is', p_dot_grad)
            c_reduct = 0.7; c_A = 1e-4; c_W = 0.9
            alpha = alpha_start = alpha_last
    #                 print('starting alpha', alpha_start)
            while validityfunc(dof+alpha*pdir, chi_invdag, Gdag, Pv)<=0: #move back into feasibility region
                alpha *= c_reduct
            alpha_feas = alpha
    #                 print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = np.inf
            while True:
                tmp_dual = justfunc(dof+alpha*pdir)[0]
                # print("tmp_dual",tmp_dual,"\n")
                # print("Dopt",Dopt,"\n")
                # print("In here bud \n")
                if tmp_dual<Dopt: #the dual is still decreasing as we backtrack, continue
                    # print("blabla")
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
    #                         print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof + alpha_feas*pdir #dof that is roughly on duality boundary
                    eig_fct_eval = mineigfunc(singular_dof, chi_invdag, Gdag, Pv) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[0]
                    mineigv=eig_fct_eval[1]
    #                         print("min_evec shape",mineigv.shape,"\n")
    #                         print("min_evec type",type(mineigv),"\n")
                            # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeSval = dgfunc(dof, np.array([]), [mineigv], tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, False, e_vac)[0]
                            # This only gets us the value of the dual function
                    epsS = np.sqrt(fakeSratio*np.abs(dualval/fakeSval))
    #                         print('epsS', epsS, '\n')
                            #fSlist.append(np.matmul(epsS,mineigv))
                    fSlist.append(epsS*mineigv) #add new fakeS to fSlist
    #                         print('length of fSlist', len(fSlist))
    #                 print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir
                    #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta, tmp_grad, fSlist, tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, True, e_vac)
            tmp_dual = tmp_val[0]
            tmp_grad = tmp_val[1]
            T=tmp_val[3]
            A=tmp_val[4]
            b=tmp_val[5]
             
            p_dot_tmp_grad = pdir @ tmp_grad
            if added_fakeS:
                Hinv = np.eye(dofnum,dtype='complex') #the objective has been modified; restart Hinv from identity
            elif p_dot_tmp_grad > c_W*p_dot_grad: #satisfy Wolfe condition, update Hinv
    #                     print('updating Hinv')
                # BFGS update
                # print("Hein?")
                gamma = tmp_grad - grad
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                # print("gamma_dot_delta ",gamma_dot_delta ,"\n")
                Hinv_dot_gamma = Hinv@tmp_grad + Ndir
    #                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
    #                     print("Hinv",Hinv,"\n")
                Hinv -= ( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta
            dualval = tmp_dual
            grad[:] = tmp_grad[:]
            dof += delta
            objval = dualval - np.dot(dof,grad)
            eqcstval = np.abs(dof) @ np.abs(grad)
    #                 print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
    #                 print('normgrad is', la.norm(grad))
            if gradConverge and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval) and la.norm(grad)<opttol * np.abs(dualval): #objective and gradient norm convergence termination
                break
            if (not gradConverge) and iternum>min_iter and np.abs(dualval-objval)<opttol*np.abs(objval) and np.abs(eqcstval)<opttol*np.abs(objval): #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            if iternum % iter_period==0:
    #                     print('prev_dualval is', prev_dualval)
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
    return dof, grad, dualval, objval, T, A, b
def item(x0, Dual, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, e_vac):
    mu=10e-8
    L=10e6
    q=mu/L
    Ak=0
    xk=x0
    yk=x0
    zk=x0
    tol = 1e-5
    cdn = False
    grad = np.zeros(len(x0))
    fSlist=[]
    indom = False
    while cdn == False and indom == False: # setup for 20 steps
        Ak=((1+q)*Ak+2*(1+math.sqrt((1+Ak)*(1+q*Ak))))/(1-q)**2
        bk=Ak/((1-q)*Ak)
        dk=((1-q**2)*Ak-(1+q)*Ak)/(2*(1+q+q*Ak))
        # store the previous xk, yk and zk values to calculate the norm
        xk_m1=xk 
        yk_m1=yk
        zk_m1=zk
        # Let's calculate the new yk 
        yk=(1-bk)*zk_m1+bk*xk_m1
        # We need to solve for |T> with the yk multipliers
        val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, True, e_vac)
        T_yk = val_yk[0] # we don't use this here but it was used to calculate the gradient below
        g_yk = val_yk[1] # this is the gradient evaluated at the |T> found with the yk multipliers
        # We can now calculate the new xk and zk values
        xk=yk-(1/L)*g_yk*yk_m1
        zk=(1-q*dk)*zk_m1+q*dk*yk_m1-(dk/L)*g_yk
        # Check if it is still necessary to go to the next iteration by
        # verifying the tolerance and if the smallest eigenvalue is positive,
        # which indicates we are in the domain. 
        if np.linalg.norm(xk-xk_m1)<tol and validityfunc(yk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True
        if np.linalg.norm(yk-yk_m1)<tol and validityfunc(xk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True
        if np.linalg.norm(zk-zk_m1)<tol and validityfunc(zk, chi_invdag, Gdag, Pv)>0:
            cdn = True
            indom = True  
    val_yk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, True, e_vac)
    D_yk = val_yk[0]
    g_yk = val_yk[1]
    obj_yk = val_yk[2]
    val_xk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, True, e_vac)
    D_xk = val_xk[0]
    g_xk = val_xk[1]
    obj_xk = val_xk[2]
    val_zk = Dual(yk, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, True, e_vac)
    D_zk = val_zk[0]
    g_zk = val_zk[1]
    obj_zk = val_zk[2]
    T = val_zk[3]
    A=val_zk[4]
    b=val_zk[5]
    dof = [yk, xk, zk]
    grad = [g_yk, g_xk, g_zk]
    dualval =[D_yk, D_xk, D_zk]
    objval = [obj_yk, obj_xk, obj_zk]
    print("results for yk, results for xk, results for zk")
    return dof, grad, dualval, objval, T, A, b
    dof, grad, dualval, objval, T, A, b
def Cpp_Execution(liste):
    # creating a pipe to child process
    data, temp = os.pipe()
    # writing inputs to stdin and using utf-8 to convert it to byte string
    # you can only write with strings: https://www.geeksforgeeks.org/python-os-write-method/
    os.write(temp, liste);
    os.close(temp)
    # temp is for writing and data is for reading
    # storing output as a byte string
    s = subprocess.check_output("g++ nomad.cpp -o out1;./out1", stdin = data, shell = True)
    # decoding to print a normal output
    a=s.decode("utf-8")
    # print(int(a)+2)
    print(s.decode("utf-8"))
def nomad(x0, dgfunc, validityfunc, mineigfunc,tsolver, ei, ei_tr, chi_invdag, Gdag, Pv, gradConverge=False, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6):
    # We need to solve for |T> with the initial multiplier values
    grad = np.zeros(len(x0))
    fSlist = [] 
    eval = Dual(x0, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=True)
    obj = eval[2][0][0]
    grad = eval[1]
    liste = [obj]
    # print("grad",grad[0])
    for i in range(len(grad)):
        liste.append(grad[i])
    # for i in grad:
    #     liste.append(grad)
    print("liste", liste, "\n")
    # The code above append to the list, which will be the input to the cpp code, in an alternating way, 
    # so it appends the value of the C1 coefficient  for some multipliers and then the value of the 
    # second constraint for the same multiplier value. This should simplify things when writing the dual 
    # function as the optimization function in the cpp code for nomad. 
    # The list has the following form:
    # liste(obj, C_1(x_0), C2_(x_0), C1(x_1), C2(x_1), ...) -> continue this for the different multiplier values x_k.
    # enter = ""
    # Driver functions
    # if __name__ == "__main__": 
    #     multipliers = Cpp_Execution(liste)
    # # Solve for the T with the new multiplier values
    # T = Dual(x0, grad, fSlist, tsolver,ei,  ei_tr, chi_invdag, Gdag, Pv, get_grad=False)[1]
    x=2
    return x
lsolvers = {
    "bfgs": BFGS_fakeS_with_restart, "item": item, "nomad": nomad} 
def lsolverpick(name,x,tsolver,ei, ei_tr,chi_invdag,Gdag, Pv, e_vac):
    return lsolvers[name](x,Dual,validityfunc, mineigfunc,tsolver,ei, ei_tr,chi_invdag,Gdag, Pv, e_vac)
