using LinearAlgebra
using SparseArrays
using IterativeSolvers

M=ones(ComplexF64, 8, 8)
M[1,1]=-0.47706242+1.24929382e-02im
M[1,2]=-0.13132647+1.20288341e-02im
M[1,3]=0.15139589+1.23379882e-02im
M[1,4]=0.00629926+1.18786237e-02im
M[1,5]=0.18577083+7.74919272e-05im
M[1,6]=-0.18577086-7.74719611e-05im
M[1,7]=-0.18577083-7.74822785e-05im
M[1,8]= 0.18577086+7.74623186e-05im

M[2,1]=-0.13132673+1.20287615e-02im
M[2,2]= -0.47706242+1.24929382e-02im
M[2,3]=0.006299  +1.18785522e-02im
M[2,4]=0.15139589+1.23379882e-02im
M[2,5]=0.0481451 +2.27716281e-04im
M[2,6]=0.18577083+7.74919272e-05im
M[2,7]=-0.0481451 -2.27687476e-04im
M[2,8]=-0.18577083-7.74822785e-05im

M[3,1]=0.15139588+1.23380041e-02im
M[3,2]=0.00629925+1.18786394e-02im
M[3,3]=-0.47706242+1.24929382e-02im
M[3,4]=-0.13132647+1.20288341e-02im
M[3,5]=0.04814507+2.27724036e-04im
M[3,6]=-0.04814517-2.27667154e-04im
M[3,7]=0.18577083+7.74919272e-05im
M[3,8]=-0.18577086-7.74719611e-05im

M[4,1]=0.00629899+1.18785678e-02im
M[4,2]=0.15139588+1.23380041e-02im
M[4,3]=-0.13132673+1.20287615e-02im
M[4,4]=-0.47706242+1.24929382e-02im
M[4,5]=0.03501468+6.71355361e-04im
M[4,6]=0.04814507+2.27724036e-04im
M[4,7]=0.0481451 +2.27716281e-04im
M[4,8]=0.18577083+7.74919272e-05im

M[5,1]=0.18577083+7.74919272e-05im
M[5,2]=0.0481451 +2.27702332e-04im
M[5,3]=0.04814507+2.27725761e-04im
M[5,4]=0.03501469+6.71320142e-04im
M[5,5]=-0.47706246+1.24932128e-02im
M[5,6]=0.15139582+1.23382785e-02im
M[5,7]=-0.13132673+1.20290725e-02im
M[5,8]= 0.00629897+1.18788765e-02im

M[6,1]=-0.18577086-7.74580125e-05im
M[6,2]=0.18577083+7.74919272e-05im
M[6,3]=-0.04814516-2.27628424e-04im
M[6,4]=0.04814507+2.27725761e-04im
M[6,5]= 0.15139588+1.23382386e-02im
M[6,6]=-0.47706246+1.24932128e-02im
M[6,7]=0.00629903+1.18788396e-02im
M[6,8]=-0.13132673+1.20290725e-02im

M[7,1]=-0.18577083-7.74840035e-05im
M[7,2]=-0.04814511-2.27678763e-04im
M[7,3]=0.18577083+7.74919272e-05im
M[7,4]=0.0481451 +2.27702332e-04im
M[7,5]=-0.13132654+1.20290697e-02im
M[7,6]=0.00629916+1.18788745e-02im
M[7,7]=-0.47706246+1.24932128e-02im
M[7,8]=0.15139582+1.23382785e-02im

M[8,1]=0.18577086+7.74500938e-05im
M[8,2]=-0.18577083-7.74840035e-05im
M[8,3]=-0.18577086-7.74580125e-05im
M[8,4]=0.18577083+7.74919272e-05im
M[8,5]= 0.00629922+1.18788376e-02im
M[8,6]=-0.13132654+1.20290697e-02im
M[8,7]= 0.15139588+1.23382386e-02im
M[8,8]=-0.47706246+1.24932128e-02im

ei=ones(ComplexF64, 8, 1)
ei[1,1]=1.01306535e-03-0.00069656im
ei[2,1]=8.65469818e-04-0.00027726im
ei[3,1]=8.65469818e-04-0.00027726im
ei[4,1]=8.27267322e-04-0.00018805im
ei[5,1]=-6.59042723e-05+0.00035215im
ei[6,1]=-7.38085739e-05+0.00020963im
ei[7,1]=-7.38085739e-05+0.00020963im
ei[8,1]=7.37869626e-05-0.00020966im

e_vac = 0.8002699193453825

s=floor(Int,sqrt(length(M)))
chi_coeff = 3.0+0.01im
chi = chi_coeff*Matrix(I, s, s)
chi_invdag = inv(conj.(transpose(chi)))
ei_tr = conj.(transpose(ei)) 
Gdag = conj.(transpose(M))
# P=I
k0=2*pi
Z=1
function C1(T) 
    # Left term
    PT=T #P*
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) # (1/2)*
    # Right term
    chiGP = (chi_invdag - Gdag)#*P
    chiG_A=(chiGP-conj.(transpose(chiGP)))/(2im)
    # M^A = (M+M^dagg)/2 -> j is i in python
    chiGA_T = chiG_A*T
    # print("transpose(T)",transpose(T),"\n")
    T_chiGA_T = real(conj.(transpose(T))*chiGA_T)
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    return I_EPT - T_chiGA_T
end
# Second constraint
function C2(T) 
    # Left term
    PT = T # P*
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT=ei_tr*PT
    I_EPT = real(EPT) # (1/2)*
    # Right term
    chiGP = (chi_invdag - Gdag)#*P
    # M^S = (M+M^dagg)/2 
    chiG_S=(chiGP+conj.(transpose(chiGP)))/2
    chiGS_T = chiG_S*T
    T_chiGS_T = real(conj.(transpose(T))*chiGS_T)
    # print("I_EPT - T_chiGS_T",I_EPT - T_chiGS_T,"\n")
    return I_EPT - T_chiGS_T
end

function Am(x) # the A matrix
    #A=zeros(ComplexF64, floor(Int, sqrt(length(chi_invdag))), floor(Int,sqrt(length(chi_invdag))))
    chiGP = (chi_invdag - Gdag)#*P
    chiG_A=(chiGP-conj.(transpose(chiGP)))/(2im)
    chiG_S=(chiGP+conj.(transpose(chiGP)))/(2)

    # print("chiGP", chiGP, "\n")
    # print("chiG_A", chiG_A, "\n")
    # print("chiG_S", chiG_S, "\n")

    A = x[1]*chiG_A + x[2]*chiG_S
    # for i in range(1,length(x),step=1)
    #     chiGP = (chi_invdag - Gdag)#*P
    #     chiG_A=(chiGP-conj!(transpose(chiGP)))/(2im)
    #     chiG_S=(chiGP+conj!(transpose(chiGP)))/(2)
    #     A = x[1]*chiG_A + x[2]*chiG_S
        # if mod(i,2) == 1 # Anti-sym
        #     chiG_A=(chiGP-conj!(transpose(chiGP)))/(2im)
        #     A = x[i]*chiG_A
        # else # Sym -> if mod(i,2) == 0
        #     chiG_S=(chiGP+conj!(transpose(chiGP)))/(2)
        #     A .+= x[i]*chiG_S
        # end
    # print("Exited A \n")
    # print("A",A,"\n")
    return A
end
function bv(x) # the b vector
    # print("In b \n")
    term1 = -1/(2.0im)
    b = term1*ei
    # print("x", length(x), "\n")
    for i in range(1,length(x),step=1) 
        if mod(i,2)==1
            term = (1/2.0im)*x[i]#*P
            b.+=term.*ei
        else
            term = (1/(2.0))*x[i]#*P
            b.+=term.*ei
        end
    end
    return b
end

# Code for the different tsolvers
function th_solve(A,b)
    T=A\b 
    return T
end
function cg_solve(A,b)
    T=cg(A,b) 
    return T
end
function bicgstab_solve(A,b)
    T= bicgstabl(A, b)
    return T
end
function gmres_solve(A,b)
    T= gmres(A, b)
    return T
end
tsolvers=Dict("th"=>th_solve, "cg"=>cg_solve, "bicgstab"=>bicgstab_solve,"gmres"=>gmres_solve)
function tsolverpick(name,A,b) 
    return tsolvers[name](A,b)
end

function Dual(x,g,fSlist, tsolver, get_grad)
    A=Am(x)
    b=bv(x)
    T=tsolverpick(tsolver,A,b)
    # print("T", T, "\n")
    g=ones(Float64, length(x), 1)
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")
    for i in range(1,length(x), step=1)
        if mod(i,2) == 1 # Asym
            g[i] = C1(T)[1]
        else   #Sym
            g[i] = C2(T)[1]
        end
        # print("g", g, "\n")
    end
    # D=0.0+0.0im
    if length(fSlist)==0
        ei_T=ei_tr*T
        obj = 0.5*(k0/Z)*imag(ei_T) .+ e_vac
        D = obj
        # print("D",D,"\n")
        for i in range(1,length(x), step=1)
            # print("x[i]*g[i]",x[i]*g[i],"\n")
            D .+= x[i]*g[i]
        end
    else
        f=fSlist[1]
        # if typeof(fSlist) == Array
        #     f=fSlist[1]
        # else # fSlist is an array
        #     f=fSlist
        # end
        # print("size(f[1])", size(f[1]), "\n")
        A_f = A*f
        f_tr = conj.(transpose(f)) 
        fAf=f_tr*A_f
        ei_T=ei_tr*T
        # print("imag(ei_T)", imag(ei_T), "\n")
        obj = 0.5*(k0/Z)*imag(ei_T)[1] + e_vac + fAf[1] 
        D = obj
        for i in range(1, length(x), step=1)
            D += x[i]*g[i]
        end
    end
    if get_grad == true
        return real(D[1]), g, real(obj), T, A, b 
    elseif get_grad == false
        return real(D[1]), real(obj), T, A, b 
    end
end

# x0 = [1,0.1]
# print("test 1: ", Dual(x0,0,[], "th", false), "\n")
# print("test 2: ", Dual(x0,0,[], "th", false), "\n")
# print("test 3: ", Dual(x0,0,[[1,2,3,4,5,6,7,8]], "th", true), "\n")
# print("test 4: ", Dual(x0,0,[[1,2,3,4,5,6,7,8]], "th", true), "\n")
    
function mineigfunc(x)
    # print("x mineigfunc",x,"\n")
    A=Am(x)
    # print("A: ",A, "\n")
    # min_eval=eigmin(A)
    # eigval = eigvals(A)
    # print("eigval", eigval, "\n")

    val = eigen(A)
    w_val=val.values
    v=val.vectors
    print("w_val", w_val, "\n")
    min_eval = w_val[1]
    # v = eigvecs(A)
    # print("v", v, "\n")
    min_evec=v[:,1]
    print("min_eval", min_eval, "\n")
    print("min_evec", min_evec, "\n")
    # print("w_val", w_val, "\n")
    # w: eigen values
    # v: eigen vectors
    # w=real(w_val)
    # min_eval=10
    # i=1
    # for element in w
    #     if 0<= element <min_eval
    #         min_eval=element
    #         i+=1
    #     end
    # end

    # print("min_eval", min_eval, "\n")
    # print("min_evec", min_evec, "\n")
    return min_eval, min_evec
end

# x0 = [1;0.1]
# value = mineigfunc(x0)

# Need to test this tomorrow!!!
function validityfunc(x)
    val=mineigfunc(x)
    min_eval=val[1]
    min_evec=val[2] # Not needed here but needed later
    print("min_eval", min_eval,"\n")
    if min_eval>0
        return 1
    else
        return -1
    end
end

function BFGS_fakeS_with_restart(initdof, dgfunc, validityfunc, mineigfunc, tsolver, gradConverge=false, opttol=1e-2, fakeSratio=1e-2, reductFactor=0.1, iter_period=20, min_iter=6)
    # print("Entered BFGS")
    dofnum = length(initdof)
    grad = zeros(Float64, dofnum, 1)
    tmp_grad = zeros(ComplexF64, dofnum, 1)
    Hinv = I #approximate inverse Hessian
    prev_dualval = Inf #+0.0im
    dof = initdof
    objval=0.0
    function justfunc(d, fSlist)
        return dgfunc(d, grad, fSlist, tsolver, false)
    end
    # justfunc = lambda d: dgfunc(d, np.array([]), fSlist, get_grad=False)
    olddualval = Inf #+0.0im
    reductCount = 0
    while true #gradual reduction of modified source amplitude, outer loop
        fSlist = [] #remove old fake sources and restart with each outer iteration
        alpha_last = 1.0 #reset step size
        Hinv = I #reset Hinv
        val = dgfunc(dof, grad, fSlist, tsolver, true)  
        dualval = val[1]
        grad = val[2]
        obj = val[3]
        T = val[4]
        A = val[5]
        b = val[6]
#             print("grad",grad,"\n")
        # This gets the value of the dual function and the dual's gradient
#             print('\n', 'Outer iteration #', reductCount, 'the starting dual value is', dualval, 'fakeSratio', fakeSratio)
        iternum = 0
        while true
            print("In 1: \n")
            # print("iternum",iternum,"\n")
            iternum += 1
#                 print('Outer iteration #', reductCount, 'Inner iteration #', iternum, flush=True)
            # print("Hinv",Hinv,"\n")
            # print("grad",grad,"\n")
            Ndir = - Hinv * grad
            # print("Ndir",Ndir,"\n")
            pdir = Ndir / norm(Ndir)
            # print("pdir", pdir, "\n")
            #backtracking line search, impose feasibility and Armijo condition
            p_dot_grad = dot(pdir, grad)
            print("grad: ", grad, "\n")
            print("pdir dot grad is: ", p_dot_grad, "\n")
            c_reduct = 0.7
            c_A = 1e-4
            c_W = 0.9
            alpha = alpha_start = alpha_last
            # print("pdir",pdir,"\n")
            # print("alpha",alpha,"\n")
            # print('starting alpha', alpha_start)
            print("validityfunc(dof.+alpha*pdir)",validityfunc(dof.+alpha.*pdir),"\n")
            print("dof", dof, "\n")
            print("alpha*pdir", alpha*pdir, "\n")
            print("alpha.*pdir", alpha.*pdir, "\n")
            while validityfunc(dof.+alpha*pdir)<=0 #move back into feasibility region
                alpha *= c_reduct
                print("alpha", alpha, "\n")
            end 
            alpha_feas = alpha
#                 print('alpha before backtracking is', alpha_feas)
            alphaopt = alpha
            Dopt = Inf #+0.0im
            while true
                # tmp_dual 1
                tmp_dual = justfunc(dof.+alpha*pdir, fSlist)[1] # , Array([]), fSlist, tsolver, false
                # print("tmp_dual",tmp_dual,"\n")
                # print("Dopt",Dopt,"\n")
                # print("In here bud \n")
                
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
            
            if Float64(alphaopt/alpha_start)>(c_reduct+1)/2 #in this case can start with bigger step
                alpha_last = alphaopt*2
                print("In 2: \n")
            else
                print("In 3: \n")
                alpha_last = alphaopt
                if alpha_feas/alpha_start < (c_reduct+1)/2 && alphaopt/alpha_feas > (c_reduct+1)/2 #this means we encountered feasibility wall and backtracking linesearch didn't reduce step size, we should add fake source
                    added_fakeS = true
#                         print('encountered feasibility wall, adding a fake source term')
                    singular_dof = dof .+ alpha_feas*pdir #dof that is roughly on duality boundary
                    print("singular_dof", singular_dof, "\n")
                    eig_fct_eval = mineigfunc(singular_dof) #find the subspace of ZTT closest to being singular to target with fake source
                    mineigw=eig_fct_eval[1]
                    mineigv=eig_fct_eval[2]
                    print("mineigw: ", mineigw, "\n")
                    print("mineigv: ", mineigv, "\n")
#                         print("min_evec shape",mineigv.shape,"\n")
#                         print("min_evec type",type(mineigv),"\n")
                    # fakeSval = dgfunc(dof, np.matrix([]), matrix, get_grad=False)
                    fakeS_eval = dgfunc(dof, Array([]), [mineigv], tsolver, false)
                    fakeSval = fakeS_eval[1]

                    # This only gets us the value of the dual function
                    print("fakeSval", fakeSval, "\n")
                    epsS = sqrt(fakeSratio*abs(dualval/fakeSval))
#                         print('epsS', epsS, '\n')
                    #fSlist.append(np.matmul(epsS,mineigv))
                    append!(fSlist,epsS*mineigv) #add new fakeS to fSlist                                  
                    #fSlist.append(epsS*mineigv) 
#                         print('length of fSlist', len(fSlist))
                end
            end
#                 print('stepsize alphaopt is', alphaopt, '\n')
            delta = alphaopt * pdir
            #########decide how to update Hinv############
            tmp_val = dgfunc(dof+delta, tmp_grad, fSlist, tsolver, true)
            tmp_dual = tmp_val[1] #tmp_dual 2
            tmp_grad = tmp_val[2]
            T = tmp_val[4]
            A = tmp_val[5]
            b = tmp_val[6]
            # Not sure what the @ does.
            p_dot_tmp_grad = dot(pdir, tmp_grad)
            if added_fakeS
                Hinv = I #the objective has been modified; restart Hinv from identity
            elseif p_dot_tmp_grad > c_W*p_dot_grad #satisfy Wolfe condition, update Hinv
#                     print('updating Hinv')
                gamma = tmp_grad - grad
                #print("gamma",gamma,"\n")
                gamma_dot_delta = alpha * (p_dot_tmp_grad-p_dot_grad)
                Hinv_dot_gamma = Hinv * tmp_grad + Ndir
#                     print("np.outer(delta, Hinv_dot_gamma)",( np.outer(Hinv_dot_gamma, delta) + np.outer(delta, Hinv_dot_gamma) - (1+np.dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*np.outer(delta, delta) ) / gamma_dot_delta,"\n")
#                     print("Hinv",Hinv,"\n")
                # print("",,"\n")
                # print("Hinv_dot_gamma*delta'",Hinv_dot_gamma*delta',"\n")
                # print("gamma_dot_delta",gamma_dot_delta,"\n")
                Hinv -= ((Hinv_dot_gamma*delta') + (delta*Hinv_dot_gamma') - (1+dot(gamma, Hinv_dot_gamma)/gamma_dot_delta)*(delta*delta') ) / gamma_dot_delta
            end
            dualval = tmp_dual
            grad[1:end] = tmp_grad[1:end]
            dof += delta
            objval = dualval .- dot(dof,grad)
            eqcstval = transpose(abs.(dof)) * abs.(grad)[1]
#                 print('at iteration #', iternum, 'the dual, objective, eqconstraint value is', dualval, objval, eqcstval)
#                 print('normgrad is', la.norm(grad))
            if gradConverge && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) && norm(grad)<opttol * abs(dualval) #objective and gradient norm convergence termination
                break
            end
            if gradConverge==true && iternum>min_iter && abs(dualval-objval)<opttol*abs(objval) && abs(eqcstval)<opttol*abs(objval) #just objective convergence termination, in this case require minimum iternum to allow for adding new constraints with 0 multiplier
                break
            end
            if mod(iternum,iter_period)==0
                # print("prev_dualval is", prev_dualval)
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
        print("dualval", dualval, "\n")
        reductCount += 1
        fakeSratio *= reductFactor #reduce the magnitude of the fake sources for the next iteration
    return dof, grad, dualval, objval
    end
end

x0 = [1;0.1]
# ones(ComplexF64, 1, 1)
#x0[2] = 0.1
# print("x0", size(x0), "\n")
tsolver="th"
sln = BFGS_fakeS_with_restart(x0, Dual, validityfunc, mineigfunc, tsolver)
print("sln", sln, "\n")

# function item(x0, Dual, validityfunc)
#     mu=10e-8
#     L=10e6
#     q=mu/L
#     Ak=0
#     xk=x0
#     yk=x0
#     zk=x0
#     tol = 1e-16
#     cdn = false
#     grad = zeros(Float64, length(x0), 1)
#     fSlist=[]
#     indom = false
#     while cdn == false && indom == false # setup for 20 steps
#         Ak=((1+q)*Ak+2*(1+sqrt((1+Ak)*(1+q*Ak))))/(1-q)^2
#         bk=Ak/((1-q)*Ak)
#         dk=((1-q^2)*Ak-(1+q)*Ak)/(2*(1+q+q*Ak))
#         # store the previous xk, yk and zk values to calculate the norm
#         xk_m1=xk 
#         yk_m1=yk
#         zk_m1=zk
#         # Let's calculate the new yk 
#         yk=(1-bk)*zk_m1+bk*xk_m1
#         # We need to solve for |T> with the yk multipliers
#         val_yk = Dual(yk, grad, fSlist, tsolver, true)
#         T_yk = val_yk[1] # we don't use this here but it was used to calculate the gradient below
#         g_yk = val_yk[2] # this is the gradient evaluated at the |T> found with the yk multipliers
#         # We can now calculate the new xk and zk values
#         xk=yk-(1/L).*g_yk.*yk_m1
#         zk=(1-q*dk)*zk_m1+q*dk*yk_m1-(dk/L).*g_yk
#         # Check if it is still necessary to go to the next iteration by
#         # verifying the tolerance and if the smallest eigenvalue is positive,
#         # which indicates we are in the domain. 
#         if norm(xk-xk_m1)<tol && validityfunc(yk)>0
#             cdn = true
#             indom = true
#         end
#         if norm(yk-yk_m1)<tol && validityfunc(xk)>0
#             cdn = true
#             indom = true
#         end
#         if norm(zk-zk_m1)<tol && validityfunc(zk)>0
#             cdn = true
#             indom = true  
#         end
#     end
#     val_yk = Dual(yk, grad, fSlist, tsolver, true)
#     D_yk = val_yk[1]
#     g_yk = val_yk[2]
#     val_xk = Dual(yk, grad, fSlist, tsolver, true)
#     D_xk = val_xk[1]
#     g_xk = val_xk[2]
#     val_zk = Dual(yk, grad, fSlist, tsolver, true) 
#     D_zk = val_zk[1]
#     g_zk = val_zk[2]
#     dof = [yk, xk, zk]
#     grad = [g_yk, g_xk, g_zk]
#     dualval = [D_yk, D_xk, D_zk]
#     print("results for yk, results for xk, results for zk")
#     return dof, grad, dualval
# end
# sln_item = item(x0, Dual, validityfunc)
# print(sln_item, "\n")

# function extracted_power_optmization(M, ei, x0, tsolver, lsolver, liste)
#     s=floor(Int,sqrt(length(M)))
#     chi_coeff = 3.0+2.0im
#     chi = chi_coeff*Matrix(I, s, s)
#     # print(typeof(I))
#     # print("conj!(chi)",conj!(chi),"\n")
#     chi_invdag = inv(conj!(transpose(chi)))
#     ei_tr = conj!(transpose(ei)) 
#     Gdag = conj!(transpose(M))
#     print("Solved using the ", tsolver, " method and the ", lsolver, " method")
#     # @time 
#     x_th=lsolverpick(lsolver, x0, tsolver, ei, ei_tr, chi_invdag, Gdag, liste)
#     gradient=x_th[end:end]
#     Dual=x_th[end-1:end-1]
#     obj=x_th[end-2:end-2]
#     print("Gradient value: ", gradient, "\n")
#     print("Objective value: ", objective, "\n")

# end 

# # Here are some global values
# Z=1 # Impedence of free space
# wvlgth = 1.0
# Qabs = 1e4
# omega = (2*pi/wvlgth) * (1+1im/2/Qabs)
# k0 = omega / 1
# dL = 0.05
# # 2 projection matrices, so 4 constraints
# # cg and bfgs
# dx=2 # -> Mx
# dy=2 # -> My 
# addx=1
# liste=[[1,1,1,1,1,1,1,1]]
# # liste=[[4,4,4,4,4,4,4,4]]
# # liste=[[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]]
# # liste=[[1,1,1,1,1,1,1,1],[5,6,7,8,1,2,3,4]]
# # liste=[[1,2,3,4,5,6,7,8],[5,6,7,8,1,2,3,4]]
# setup = DipoleOverSlab_setup(dx,dy,addx,liste,dL, wvlgth)
# print("setup",setup)
# M=setup[1]
# # print("M shape",M.shape,"\n")
# ei = setup[2]
# print("ei", ei,"\n")
# Pv = setup[3]
# e_vac=setup[4]
# print("e_vac",e_vac,"\n")
# x0 = ones(ComplexF64, 2*length(liste), 1)
# # x0[0] = 1
# # x0[1] = 1.1
# tsolver = "th"
# lsolver = "bfgs"
# bfgs_time = []
# opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv, e_vac)
# dual = opt[1]
# T = opt[2]


