module bfgs_power_iteration_asym_only
export mineigfunc, validityfunc, BFGS_fakeS_with_restart_pi, power_iteration_first_evaluation, power_iteration_second_evaluation
using LinearAlgebra, G_v_product
# BFGS with restart code
# This file also contains code for the minimum eigenvalue
# calculation and a function that verifies if we are
# still in the domain of duality
function power_iteration_first_evaluation(l,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    # Ideally choose a random vector
    # to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    # let's implement an error check using the ratios of different iterations 
    good = false 
    while good == false
        # calculate the n+1 term
        #calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = output(l,b_k,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # A*b_k
        # calculate the norm
        A_bk_norm = norm(A_bk)
        # re normalize the vector
        b_k1 = A_bk / A_bk_norm

        # calculate the n+2 term
        # G|v> type calculation
        A_bk1 = output(l,b_k1,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # A*b_k using the new b_k (the n+1 one)
        # calculate the norm
        A_bk1_norm = norm(A_bk1)
        # re normalize the vector
        b_k = A_bk1 / A_bk1_norm # technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = output(l,b_k,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) 

        # calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(A_bk1)/norm(A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(A_bk2)/norm(A_bk1)
        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
            # print("Good \n")
        end
        # print("ratio_n1_n ", ratio_n1_n, "\n")
        # print("ratio_n2_n1 ", ratio_n2_n1, "\n")
        # print("abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n ", abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n, "\n")
        # print("good ", good, "\n")
        global eigvector = b_k
        global A_eigvector = A_bk2
    end
    # print("Done \n")

    # it is better to implement an error check than a for loop using the number of iterations like below
    # input and output ratio 
    # norm(A^{n+1}*x)/norm(A^{n}*x) -> for a couple in a row -> ratio of the ratios
    # abs({n+2/n+1}-{n+1/n})/{n+1/n}
    # new ratio at each iteration 

    # the last b_k is the largest eigenvector of the linear operator A
    # calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    #A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]
    # print("b_k first ", eigvector, "\n")
    # print("eigenvalue first ", eigenvalue, "\n")

    return b_k, eigenvalue
end

function power_iteration_second_evaluation(l,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    # Ideally choose a random vector
    # to decrease the chance that our vector
    # is orthogonal to the eigenvector
    b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    # let's do the first run of the power iteration method to get an intial largest eigenvalue
    # and corresponding largest eigenvector
    evaluation = power_iteration_first_evaluation(l,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    eigvect_1 = evaluation[1] 
    eigval_1 = evaluation[2] 
    # let's define some values
    # chi coefficient
    chi_coeff = 3.0 + 0.01im
    # inverse chi coefficient
    chi_inv_coeff = 1/chi_coeff 
    chi_inv_dag_coeff = conj(chi_inv_coeff)
    # define the projection operators
    # I didn't include P in the calculation since P = I for this case
    b_k = rand(ComplexF64, 3*cellsA[1]*cellsA[2]*cellsA[3], 1)
    good = false 
    while good == false
        # calculate the n+1 term
        # calculate the operator-vector product (A linear operator and b_k vector)
        # G|v> type calculation
        A_bk = output(l,b_k,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # A|v> calculation 
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk = eigval_1*b_k .- A_bk # this the A_v>=A*b_k calculation
        # calculate the norm
        eigval_A_bk_norm = norm(eigval_A_bk)
        # re normalize the vector
        b_k1 = eigval_A_bk / eigval_A_bk_norm # this is the n+1 term

        # calculate the n+2 term
        # G|v> type calculation
        A_bk1 = output(l,b_k1,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # A*b_k using the new b_k (the n+1 one)
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk1 = eigval_1*b_k1 .- A_bk1 # this the A_v>=A*b_k calculation
        # calculate the norm
        eigval_A_bk1_norm = norm(eigval_A_bk1)
        # re normalize the vector
        b_k = eigval_A_bk1 / eigval_A_bk1_norm # this is the n+2 term
        # technically b_k2 but it's called b_k since it will be the
        # b_k for the next iteration of the loop

        # calculate the A*b_k product with the new b_k (the n+2 one)
        # G|v> type calculation
        A_bk2 = output(l,b_k,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) 
        # let's do the (eigval*I-A)v> = eigval*|v>-A*|v> calculation
        eigval_A_bk2 = eigval_1*b_k .- A_bk2 # this the A_v>=A*b_k calculation

        # calculate the ratios 
        # norm(A^{n+1}*x)/norm(A^{n}*x)
        ratio_n1_n = norm(eigval_A_bk1)/norm(eigval_A_bk)
        # norm(A^{n+2}*x)/norm(A^{n+1}*x)
        ratio_n2_n1 = norm(eigval_A_bk2)/norm(eigval_A_bk1)
        if abs(ratio_n2_n1-ratio_n1_n)/ratio_n1_n < 0.01
            good = true
            # print("Good \n")
        end
        global eigvector = b_k
        global A_eigvector = eigval_A_bk2
    end

    # the last b_k is the largest eigenvector of the linear operator A for the second run 
    # of the power iteration method
    # calculate the eigenvalue corresponding to the largest eigenvector b_k 
    # using the formula: eigenvalue = (Ax * x)/(x*x), where x=b_k in our case
    # A*x=A*b_k is a G|v> type calculation
    #A_bk = output(l,b_k,cellsA)
    A_bk2_conj_tr = conj.(transpose(A_eigvector)) 
    bk_conj_tr = conj.(transpose(eigvector)) 
    eigenvalue_2 = real((A_bk2_conj_tr*eigvector)/(bk_conj_tr*eigvector))[1]

    # print("b_k second ", eigvector, "\n")
    # print("eigenvalue second ", eigenvalue_2, "\n")


    # the minimum eigenvalue will be found by substracting the largest eigenvalue of the second run 
    # and that of the first run => lambda_2-lambda_1
    # will the minimum eigenvector be found by substracting the two eigenvectors?
    min_eigval = eigenvalue_2 - eigval_1
    min_eigvec = eigvector - eigvect_1
    print("min_eigval ", min_eigval, "\n")
    return min_eigvec, min_eigval
end

function validityfunc(l,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    eval_2 = power_iteration_second_evaluation(l,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    
    min_eigvec = eval_2[1] # Not needed here but needed later
    min_eigval = eval_2[2]
    # val=mineigfunc(x)
    # min_eval=val[1]
    # min_evec=val[2] # Not needed here but needed later
    # print("min_eval validityfunc ", min_eval,"\n")
    if min_eigval>0
        return 1
    else
        return -1
    end
end
# mineigfunc is the power_iteration_second_evaluation function
# l = initdof
function BFGS_fakeS_with_restart_pi(l,dgfunc,P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,validityfunc, mineigfunc, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    # print("Entered BFGS")
    dofnum = length(l)
    dualval = 0.0
    grad = zeros(Float64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf 
    dof = l
    objval=0.0
    function justfunc(d,fSlist)
        return dgfunc(d,Array([]),P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,fSlist,false)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf #+0.0im
    reductCount = 0
    while true #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(dof,grad,P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,fSlist,true)  
        dualval = val[1]
        grad = val[2]
        obj = val[3]
        # T = val[4]
        # A = val[5]
        # b = val[6]
#             print("grad",grad,"\n")
        # This gets the value of the dual function and the dual's gradient
        print('\n', "Outer iteration # ", reductCount, " the starting dual value is ", dualval, " fakeSratio ", fakeSratio, "\n")
        iternum = 0
        while true
            # print("In 1: \n")
            # print("iternum",iternum,"\n")
            iternum += 1
            print("Outer iteration # ", reductCount, " Inner iteration # ", iternum, "\n")
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
            print("starting alpha ", alpha_start, "\n")
            # print("validityfunc(dof.+alpha*pdir)",validityfunc(dof.+alpha.*pdir),"\n")
            # print("dof", dof, "\n")
            # print("alpha*pdir", alpha*pdir, "\n")
            # print("alpha.*pdir", alpha.*pdir, "\n")
            print("eigval ", validityfunc(dof.+alpha*pdir,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
            , "\n")
            while validityfunc(dof.+alpha*pdir,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)<=0 #move back into feasibility region
                alpha *= c_reduct
                print("alpha ", alpha, "\n")
            end 
            alpha_feas = alpha
            print("alpha before backtracking is ", alpha_feas, "\n")
            alphaopt = alpha
            Dopt = Inf #+0.0im
            while true
                # tmp_dual 1
                # print("dof.+alpha*pdir ",dof.+alpha*pdir.+[1],"\n")
                # print("fSlist ", fSlist,"\n")
                tmp_dual = justfunc(dof.+alpha*pdir, fSlist)[1] # , Array([]), fSlist, tsolver, false
                print("tmp_dual",tmp_dual,"\n")
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
                    print("encountered feasibility wall, adding a fake source term \n")
                    singular_dof = dof .+ alpha_feas*pdir #dof that is roughly on duality boundary
                    print("singular_dof", singular_dof, "\n")
                    eig_fct_eval = mineigfunc(singular_dof,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigv=eig_fct_eval[1] # eigenvector 
                    mineigw=eig_fct_eval[2] # eigenvalue
                    # print("mineigw: ", mineigw, "\n")
                    # print("mineigv: ", mineigv, "\n")
#                         print("min_evec shape",mineigv.shape,"\n")
#                         print("min_evec type",type(mineigv),"\n")
                    # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeS_eval = dgfunc(dof,Array([]),P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,[mineigv],false)
                    fakeSval = fakeS_eval[1]
                    # This only gets us the value of the dual function
                    # print("fakeSval", fakeSval, "\n")
                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
#                         print('epsS', epsS, '\n')
                    #fSlist.append(np.matmul(epsS,mineigv))
                    print("Appending to fSlist \n")
                    append!(fSlist,epsS*[mineigv]) #add new fakeS to fSlist                                  
                    #fSlist.append(epsS*mineigv) 
                    print("length of fSlist ", length(fSlist), "\n")
                    # print("fSlist", fSlist, "\n")
                end
            end
            print("Done \n")
            print("stepsize alphaopt is ", alphaopt, '\n')
            delta = alphaopt * pdir
            print("delta ", delta, "\n")
            #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta,tmp_grad,P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,fSlist, true)
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
                print("updating Hinv \n")
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
            print("dualval ", dualval, "\n")
            objval = dualval - dot(dof,grad)
            print("objval ", objval, "\n")
            eqcstval = dot(abs.(dof),abs.(grad))
            print("eqcstval ", eqcstval, "\n")
            print("at iteration # ", iternum, " the dual, objective, eqconstraint value is ", dualval,"  " ,objval,"  " , eqcstval, "\n")
            print("normgrad is ", norm(grad), "\n")
            if gradConverge==false && iternum>min_iter && abs(abs(dualval)-abs(objval))<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
                print("First condition is attained \n")
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
                print("Second condition is attained \n")
            end
            if mod(iternum,iter_period)==0
                print("prev_dualval is ", prev_dualval, "\n")
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