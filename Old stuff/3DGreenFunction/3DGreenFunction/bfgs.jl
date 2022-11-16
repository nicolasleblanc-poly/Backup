module bfgs
export mineigfunc, validityfunc, BFGS_fakeS_with_restart
using A_lin_op, dual_func
# BFGS with restart code
# This file also contains code for the minimum eigenvalue
# calculation and a function that verifies if we are
# still in the domain of duality
function mineigfunc(x)
    A=Am(x)
    val = eigen(A)
    w_val=val.values
    v=val.vectors
    min_eval = w_val[1]
    min_evec=v[:,1]
    return min_eval, min_evec
end
function validityfunc(x)
    val=mineigfunc(x)
    min_eval=val[1]
    # min_evec=val[2] # Not needed here but needed later
    # print("min_eval validityfunc ", min_eval,"\n")
    if min_eval>0
        return 1
    else
        return -1
    end
end

function BFGS_fakeS_with_restart(initdof,dgfunc,ei,e_vac,asym,sym,cellsA,validityfunc, mineigfunc, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    # print("Entered BFGS")
    dofnum = length(initdof)
    dualval = 0.0
    grad = zeros(Float64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf 
    dof = initdof
    objval=0.0
    function justfunc(d,fSlist)
        return dgfunc(d,Array([]),l,g,P,ei,e_vac,asym,sym,cellsA,fSlist,get_grad)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf #+0.0im
    reductCount = 0
    while true #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(dof,grad,l,g,P,ei,e_vac,asym,sym,cellsA,fSlist,get_grad)  
        dualval = val[1]
        grad = val[2]
        obj = val[3]
        # T = val[4]
        # A = val[5]
        # b = val[6]
#             print("grad",grad,"\n")
        # This gets the value of the dual function and the dual's gradient
        print('\n', "Outer iteration # ", reductCount, " the starting dual value is ", dualval, " fakeSratio ", fakeSratio)
        iternum = 0
        while true
            # print("In 1: \n")
            # print("iternum",iternum,"\n")
            iternum += 1
            print("Outer iteration # ", reductCount, " Inner iteration # ", iternum)
            # print("Hinv",Hinv,"\n")
            # print("grad",grad,"\n")
            Ndir = - Hinv * grad
            # print("Ndir",Ndir,"\n")
            pdir = Ndir / norm(Ndir)
            # print("pdir", pdir, "\n")
            #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = dot(pdir, grad)
            # print("grad: ", grad, "\n")
            print("pdir dot grad is: ", p_dot_grad, "\n")
            c_reduct = 0.7
            c_A = 1e-4
            c_W = 0.9
            alpha = alpha_start = alpha_last
            # print("pdir",pdir,"\n")
            # print("alpha",alpha,"\n")
            print("starting alpha", alpha_start)
            # print("validityfunc(dof.+alpha*pdir)",validityfunc(dof.+alpha.*pdir),"\n")
            # print("dof", dof, "\n")
            # print("alpha*pdir", alpha*pdir, "\n")
            # print("alpha.*pdir", alpha.*pdir, "\n")
            while validityfunc(dof.+alpha*pdir)<=0 #move back into feasibility region
                alpha *= c_reduct
                print("alpha", alpha, "\n")
            end 
            alpha_feas = alpha
            print("alpha before backtracking is", alpha_feas)
            alphaopt = alpha
            Dopt = Inf #+0.0im
            while true
                # tmp_dual 1
                tmp_dual = justfunc(dof.+alpha*pdir, fSlist)[1] # , Array([]), fSlist, tsolver, false
                # print("tmp_dual",tmp_dual,"\n")
                # print("Dopt",Dopt,"\n")
                if tmp_dual < Dopt #the dual is still decreasing as we backtrack, continue
                    Dopt = tmp_dual
                    alphaopt=alpha
                else
                    alphaopt=alpha ###ISSUE!!!!
                    break
                end
                if tmp_dual <= dualval + c_A*alpha*p_dot_grad #Armijo backtracking condition
                    alphaopt = alpha
                    break
                end                      
                alpha *= c_reduct
            end
            added_fakeS = false
            print("alphaopt/alpha_start: ", alphaopt/alpha_start, "\n")
            print("(c_reduct+1)/2: ", (c_reduct+1)/2, "\n")
            if alphaopt/alpha_start>(c_reduct+1)/2 #in this case can start with bigger step
                alpha_last = alphaopt*2
                # print("In 2: \n")
            else
                #print("In 3: \n")
                alpha_last = alphaopt
                if alpha_feas/alpha_start < (c_reduct+1)/2 && alphaopt/alpha_feas > (c_reduct+1)/2 #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = true
                    print("encountered feasibility wall, adding a fake source term")
                    singular_dof = dof .+ alpha_feas*pdir #dof that is roughly on duality boundary
                    print("singular_dof", singular_dof, "\n")
                    eig_fct_eval = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[1]
                    mineigv=eig_fct_eval[2]
                    # print("mineigw: ", mineigw, "\n")
                    # print("mineigv: ", mineigv, "\n")
#                         print("min_evec shape",mineigv.shape,"\n")
#                         print("min_evec type",type(mineigv),"\n")
                    # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeS_eval = dgfunc(dof,Array([]),P,ei,e_vac,asym,sym,cellsA,[mineigv],false)
                    fakeSval = fakeS_eval[1]
                    # This only gets us the value of the dual function
                    # print("fakeSval", fakeSval, "\n")
                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
#                         print('epsS', epsS, '\n')
                    #fSlist.append(np.matmul(epsS,mineigv))
                    append!(fSlist,epsS*[mineigv]) #add new fakeS to fSlist                                  
                    #fSlist.append(epsS*mineigv) 
                    print("length of fSlist", length(fSlist), "\n")
                    print("fSlist", fSlist, "\n")
                end
            end
            print("stepsize alphaopt is", alphaopt, '\n')
            delta = alphaopt * pdir
            print("delta", delta, "\n")
            #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta,tmp_grad,P,ei,e_vac,asym,sym,cellsA,fSlist, true)
            tmp_dual = tmp_val[1] #tmp_dual 2
            tmp_grad = tmp_val[2]
            # T = tmp_val[4]
            # A = tmp_val[5]
            # b = tmp_val[6]
            p_dot_tmp_grad = dot(pdir, tmp_grad)
            if added_fakeS == true
                # print("Entered \n")
                Hinv = I #the objective has been modified; restart Hinv from identity
            elseif p_dot_tmp_grad > c_W*p_dot_grad #satisfy Wolfe condition, update Hinv
                print("updating Hinv")
                gamma = tmp_grad - grad
                #print("gamma",gamma,"\n")
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv * tmp_grad + Ndir
#                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
#                     print("Hinv",Hinv,"\n")
                # print("",,"\n")
                # print("Hinv_dot_gamma*delta'",Hinv_dot_gamma*delta',"\n")
                # print("gamma_dot_delta",gamma_dot_delta,"\n")
                Hinv -= ((Hinv_dot_gamma.*delta') + (delta.*Hinv_dot_gamma') - (1+dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*(delta.*delta') ) / gamma_dot_delta
                print("Hinv2: ", Hinv, "\n")
            end
            dualval = tmp_dual
            grad[1:end] = tmp_grad[1:end]
            dof += delta
            print("delta", delta, "\n")
            print("dualval", dualval, "\n")
            objval = dualval - dot(dof,grad)
            eqcstval = dot(abs.(dof),abs.(grad))
            print("eqcstval ", eqcstval, "\n")
            print("at iteration #", iternum, "the dual, objective, eqconstraint value is", dualval, objval, eqcstval, "\n")
            print("normgrad is", norm(grad), "\n")
            if gradConverge==false && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            end
            if mod(iternum,iter_period)==0
                print("prev_dualval is", prev_dualval)
                # print("dual val", dualval, "\n")
                if abs(prev_dualval.-dualval)<abs(dualval)*opttol #dual convergence / stuck optimization termination
                    break
                end
                prev_dualval = dualval
            end
        end
        if abs(olddualval-dualval)<opttol*abs(dualval)
            break
        end
        """
        #if len(fSlist)<=1 and np.abs(olddualval-dualval)<opttol*np.abs(dualval): #converged w.r.t. reducing fake sources
            #break
        # if len(fSlist)==0 #converged without needing fake sources, suggests strong duality
        #     break
        """
        olddualval = dualval
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    end
    return dof, grad, dualval, objval
end
end