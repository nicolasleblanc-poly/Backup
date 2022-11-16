module dual_asym_only
export dual, c1
using A_asym_only, b_asym_only, gmres, G_v_product, v_G_product
# Code to get the value of the objective and of the dual.
# The code also calculates the constraints
# and can return the gradient.
function c1(l,P,ei,T,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA) # asymmetric part 
    # Left term
    # print("In c1 \n")
    PT=P*T 
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) 
    # Right term => asym*T
    # G|v> type calculation
    g_output = output(g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,T,cellsA)
    o_x = g_output[1] # total x output 
        o_x_dag = g_output[2] # total x dag output  
    # let's define some values
    # chi coefficient
    chi_coeff = 3.0 + 0.01im
    # inverse chi coefficient
    chi_inv_coeff = 1/chi_coeff 
    chi_inv_dag_coeff = conj(chi_inv_coeff)
    # define the projection operators
    # I didn't include P in the calculation since P = I for this case
    term1 = chi_inv_dag_coeff*T
    term2 = o_x_dag
    term3 = chi_inv_coeff*T
    term4 = o_x
    asym_T = l[1]*(1\(2im))*(term1-term2-term3+term4)

    # asym_T = G_v_product(asym, T, cellsA)

    T_asym_T = conj.(transpose(T))*asym_T
    print("I_EPT ",I_EPT,"\n")
    print("T_asym_T ", T_asym_T,"\n")
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    return real(I_EPT - T_asym_T)[1]
end
function dual(l,g,P,ei,e_vac,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA,fSlist,get_grad)
    b = bv(ei, l, P) 
    T=GMRES_with_restart_x(l,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag, b, cellsA)
    g=ones(Float64, length(l), 1)
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")
    g[1] = c1(l,P,ei,T,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA)
    # print("ei ", ei, "\n")
    ei_tr = conj.(transpose(ei)) 
    k0 = 2*pi
    Z = 1

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
    print("Done dual \n")
end
end