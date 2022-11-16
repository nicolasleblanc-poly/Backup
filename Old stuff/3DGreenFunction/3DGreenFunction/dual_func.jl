module dual_func 
export dual, c1, c2
using A_lin_op, b_vector, gmres, G_v_product, v_G_product
# Code to get the value of the objective and of the dual.
# The code also calculates the constraints
# and can return the gradient.
function c1(P,ei,T,asym,cellsA) # asymmetric part 
    # Left term
    PT=P*T 
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) 
    # Right term
    # asym*T
    asym_T = G_v_product(asym, T, cellsA)
    # print("transpose(T)",transpose(T),"\n")
    T_asym_T = real(conj.(transpose(T))*asym_T)
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    return I_EPT - T_asym_T
end
# Second constraint
function c2(P,ei,T,sym,cellsA) # symmetric part 
    # Left term
    PT=P*T 
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = real(EPT) 
    # Right term
    sym_T = G_v_product(sym, T, cellsA)
    T_sym_T = real(conj.(transpose(T))*sym_T)
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    return I_EPT - T_sym_T
end
function dual(l,g,P,ei,e_vac,asym,sym,cellsA,fSlist,get_grad)
    A = A_op(l,asym,sym)
    b = bv(ei, l, P) 
    T=GMRES_with_restart(A,b,cellsA)
    g=ones(Float64, length(l), 1)
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")
    g[1] = c1(P,ei,T,asym,cellsA)
    g[2] = c2(P,ei,T,sym,cellsA)
    if length(fSlist)==0
        ei_T=ei_tr*T
        obj = 0.5*(k0/Z)*imag(ei_T) .+ e_vac
        D = obj
        for i in range(1,length(l), step=1)
            D .+= l[i]*g[i]
        end
    else
        fSval = 0
        for k in fSlist
            A_k = A*k
            k_tr = conj.(transpose(k)) 
            kAk=k_tr*A_k
            fSval += real(kAk[1])
        end
        # print("value ", fSval, "\n")
        ei_T=ei_tr*T
        # print("imag(ei_T)", imag(ei_T), "\n")
        # obj = 0.5*(k0/Z)*imag(ei_T)[1] + e_vac + fAf[1] 
        obj = 0.5*(k0/Z)*imag(ei_T)[1] + e_vac + fSval
        D = obj
        for i in range(1, length(l), step=1)
            D += l[i]*g[i]
        end
    end
    if get_grad == true
        return real(D[1]), g, real(obj) # , T, A, b 
    elseif get_grad == false
        return real(D[1]), real(obj) # , T, A, b 
    end
end
end