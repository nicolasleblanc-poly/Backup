module dual_asym_only
export dual, c1
using A_asym_only, b_asym_only, gmres, G_v_product, v_G_product
# Code to get the value of the objective and of the dual.
# The code also calculates the constraints
# and can return the gradient.
function c1(l,P,ei,T,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # asymmetric part 
    # Left term
    # print("In c1 \n")
    PT=P*T 
    # ei_tr = transpose(ei) # we have <ei^*| instead of <ei|
    ei_tr = conj.(transpose(ei))
    EPT=ei_tr*PT
    I_EPT = imag(EPT) 
    # print("T ", T, "\n")
    # Right term => asym*T
    # G|v> type calculation
    asym_T = output(l,T,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)/l[1]
    # print("asym_T", asym_T, "\n")
    T_asym_T = conj.(transpose(T))*asym_T
    print("T_asym_T ", T_asym_T*l[1], "\n")
    # print("I_EPT ", I_EPT,"\n")
    # print("T_asym_T ", T_asym_T,"\n")
    # print("I_EPT - T_chiGA_T",I_EPT - T_chiGA_T,"\n")
    # return real(I_EPT - T_asym_T[1]) # for the <ei^*| case
    return real(I_EPT - T_asym_T)[1] 
end
function dual(l,g,P,ei,e_vac, fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,fSlist,get_grad)
    b = bv_asym_only(ei, l, P) 
    # print("l ", l, "\n")
    # print("b ", b, "\n")
    l = [2] # initial Lagrange multipliers
    T=GMRES_with_restart(l,b,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    g=ones(Float64, length(l), 1)
    # print("C1(T)", C1(T)[1], "\n")
    # print("C2(T)", C2(T)[1], "\n")
    g[1] = c1(l,P,ei,T,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
    # print("g[1] ", g[1], "\n")
    # print("ei ", ei, "\n")
    # ei_tr = transpose(ei)
    ei_tr = conj.(transpose(ei)) 
    k0 = 2*pi
    Z = 1
    # I put the code below here since it is used no matter the lenght of fSlist
    ei_T=ei_tr*T
    obj = 0.5*(k0/Z)*imag(ei_T)[1] + e_vac # this is just the objective part of the dual
    D = obj
    for i in range(1,length(l), step=1)
        D += l[i]*g[i]
    end
    if length(fSlist)>0
        fSval = 0
        for k in fSlist
            A_k = output(l,k,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
            k_tr = conj.(transpose(k)) 
            kAk=k_tr*A_k
            fSval += real(kAk[1])
        end
        D += fSval
    end
    # print("Done dual \n")
    if get_grad == true
        return real(D[1]), g, real(obj) 
    elseif get_grad == false
        return real(D[1]), real(obj) 
    end
end
end