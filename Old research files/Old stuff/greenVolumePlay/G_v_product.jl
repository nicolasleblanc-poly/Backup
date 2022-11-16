module G_v_product
export G_v, output
using Embedding, Projection, FFTW, fft

"""
    Use dfft function for the direct fft 
    Use ifft function for the inverse fft
    These two functions can be found in the fft.jl file
"""

# Tests to do:
# inner product aka multiplying a vector with the complex conjugate transpose of itself
# of the result of output on v with complex conjugate transpose of v
# the number should be postive if the operator is positive definite

# 2 forms of the dual
# obj + lambdas * constraints
# <T|A|T> is an alternate form of the dual 
# see if the results are similar  

# projection and embeddeding are done for scalar fields 

function output(l,v,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA) # add y- and z-direction
    # let's define some values
    dim = cellsA[1]*cellsA[2]*cellsA[3]
    o_new = Array{ComplexF64}(undef,3*dim,1) # things are stored in ei like: [v_new_x, v_new_y, v_new_z]
    # chi coefficient
    chi_coeff = 3.0 + 0.01im
    # inverse chi coefficient
    chi_inv_coeff = 1/chi_coeff 
    chi_inv_dag_coeff = conj(chi_inv_coeff)
    # define the projection operators
    # I didn't include P in the calculation since P = I for this case
    
    # x-direction
    o_xx = G_v(g_xx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # print("o_xx ", o_xx, "\n")
    # xy ouput
    o_xy = G_v(g_xy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xz ouput
    o_xz = G_v(g_xz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total output 
    o_x = o_xx + o_xy + o_xz 
    # new code
    # xx ouput 
    o_xx_dag = Gdag_v(g_xx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xy ouput
    o_xy_dag = Gdag_v(g_xy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xz ouput
    o_xz_dag = Gdag_v(g_xz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total dag output
    o_x_dag = o_xx_dag  + o_xy_dag + o_xz_dag 
    # let's calculate the result of the calculation 
    o_new_x =  o_x  -o_x_dag 

    # # v_new_x = l[1]*(1\(2im))*(chi_inv_dag_coeff*v.-o_x_dag.-chi_inv_coeff*v.+o_x) #A and b_k product
    o_new[1:dim] = o_new_x[:]

    # # print("length(term1_x) ", length(chi_inv_dag_coeff*v), "\n")
    # # print("length(term2_x) ", length(o_x_dag), "\n")
    # # print("length(term3_x) ", length(chi_inv_coeff*v), "\n")
    # # print("length(term4_x) ", length(o_x), "\n")

    # y-direction
    o_yx = G_v(g_yx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xy ouput
    o_yy = G_v(g_yy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xz ouput
    o_yz = G_v(g_yz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total output 
    o_y =  o_yy  + o_yx + o_yz 
    # new code
    # xx ouput 
    o_yx_dag = Gdag_v(g_yx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xy ouput
    o_yy_dag = Gdag_v(g_yy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xz ouput
    o_yz_dag = Gdag_v(g_yz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total dag output
    o_y_dag = o_yx_dag  + o_yy_dag + o_yz_dag 
    # let's calculate the result of the calculation 
    o_new_y =   o_y - o_y_dag
    # v_new_y = l[1]*(1\(2im))*(chi_inv_dag_coeff*v.-o_y_dag.-chi_inv_coeff*v.+o_y) #A and b_k product
    o_new[dim+1:2*dim] = o_new_y[:]
    # print("length(term1_x) ", length(chi_inv_dag_coeff*v), "\n")
    # print("length(term2_x) ", length(o_x_dag), "\n")
    # print("length(term3_x) ", length(chi_inv_coeff*v), "\n")
    # print("length(term4_x) ", length(o_x), "\n")

    # z-direction
    o_zx = G_v(g_zx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # zy ouput
    o_zy = G_v(g_zy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # zz ouput
    o_zz = G_v(g_zz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total output 
    o_z =  o_zz  + o_zx + o_zy 
    # new code
    # xx ouput # keep the [1:8] and other parts.
    o_zx_dag = Gdag_v(g_zx, v[1:8], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xy ouput
    o_zy_dag = Gdag_v(g_zy, v[9:16], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # xz ouput
    o_zz_dag = Gdag_v(g_zz, v[17:24], fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA)
    # x total dag output
    o_z_dag = o_zz_dag  + o_zx_dag + o_zy_dag 
    # let's calculate the result of the calculation 
    o_new_z =  o_z  -o_z_dag 
    # v_new_z = l[1]*(1\(2im))*(chi_inv_dag_coeff*v.-o_z_dag.-chi_inv_coeff*v.+o_z) #A and b_k product
    # v_new_z = l[1]*(1\(2im))*(term1_z.-term2_z.-term3_z.+term4_z) #A and b_k product
    o_new[2*dim+1:3*dim] = o_new_z[:]

    # # print("length(term1_x) ", length((chi_inv_dag_coeff - chi_inv_coeff)*v), "\n")
    # # print("length(term2_x) ", length(o_new), "\n")

    # test
    v_new = l[1]*(o_new .+ (10im)*v)
    # end of test 
    # v_new = l[1]*(1\(2im))*(o_new .+ (10im)*v)
    # v_new = l[1]*(1\(2im))*(o_new .+ (chi_inv_dag_coeff - chi_inv_coeff)*v)
    # print("o_new ", o_new, "\n")
    # print("(chi_inv_dag_coeff - chi_inv_coeff)*v ", (chi_inv_dag_coeff - chi_inv_coeff)*v, "\n")
    # print("o_new .+ (chi_inv_dag_coeff - chi_inv_coeff)*v ", o_new .+ (chi_inv_dag_coeff - chi_inv_coeff)*v, "\n")
    # print("l[1] ", l[1], "\n")
    # print("v_new ", v_new, "\n")

    # replace chi_inv_dag_coeff - chi_inv_coeff by + 10im
    # test if output defines a positive definite matrix (aka all entries have a positive imaginary part)
    # if true, then test between the two different values of the dual -> the dual should be positive
    
    return v_new # v_new # o_new
end

# changes 
# didn't consider o_dag_...
# o_new as output instead of v_new 

# G should be positive definied when used in multiplication (aka have all imaginary parts)

# function for G|v>, where |v> is a vector 
# code for the AA case 
function G_v(g,v,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA) # add cellsB for the BA case 
    # 1. embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
    reshaped_v_embedded = reshape(v_embedded, (2*cellsA[1], 2*cellsA[2], 2*cellsA[3])) # v is now an array
    # 2. fft of the embedded reshaped v and of g (aka the Green function)
    
    """
    Use dfft function for the direct fft 
    Use ifft function for the inverse fft
    These two functions can be found in the fft.jl file
    """
    
    fft_reshaped_v_embedded = fft_plan_z*(fft_plan_y*(fft_plan_x*reshaped_v_embedded)) # how to do x, then y and then z 
    fft_g = fft_plan_z*(fft_plan_y*(fft_plan_x*g)) # how to do x, then y and then z 
    # print("fft_g ", fft_g, "\n")
    # 3. element-wise multiplication of g and v
    mult = fft_g .* fft_reshaped_v_embedded
    # 4. inverse fft of the multiplication of g and v 
    fft_inv = inv_fft_plan_x*(inv_fft_plan_y*(inv_fft_plan_z*mult)) # how to do z, then y and then x 
    # let's turn the previous result, which is an array, into a vector
    fft_inv_vect = reshape(fft_inv, (2*cellsA[1]*2*cellsA[2]*2*cellsA[3],1))
    # 5. project the fft inverse 
    projection = proj(fft_inv_vect, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
    # 6. add small complex offset
    result = G_offset(v) + projection
    return result # projection 
end

function G_offset(v)
    return (0.001im)*v
end

# function for Gdag|v>, where |v> is a vector 
# when working with gdag, the order of the fft 
# and the inverse fft are swapped. 
function Gdag_v(g, v, fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,cellsA) 
    # calculate gdag 
    # the code below has the same effect as calculating gddag, 
    # which would normally be done like: gdag = conj.(transpose(g)) 
    # embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
    reshaped_v_embedded = reshape(v_embedded, (2*cellsA[1], 2*cellsA[2], 2*cellsA[3])) # v is now an array
    # inverse fft of the embedded v and of gdag
    # calculate the direct fft of g_dag 
    fft_g = conj.(fft_plan_z*(fft_plan_y*(fft_plan_x*g))) # how to do x, then y and then z 
    # calculate the inverse fft of the embedded vector
    fft_inv_reshaped_v_embedded = inv_fft_plan_x*(inv_fft_plan_y*(inv_fft_plan_z*reshaped_v_embedded))
    # 3. element-wise multiplication of g and v
    mult = fft_g .* fft_inv_reshaped_v_embedded
    # 4. fft of the multiplication of g and v 
    # direct fft of the product
    fft_prod = fft_plan_z*(fft_plan_y*(fft_plan_x*mult))
    # let's turn the previous result, which is an array, into a vector
    fft_prod_vect = reshape(fft_prod, (2*cellsA[1]*2*cellsA[2]*2*cellsA[3],1))
    # 5. project the fft inverse 
    projection = proj(fft_prod_vect, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
    # 6. add small complex offset
    dag_result = G_offset(v) + projection
    return dag_result # projection 
end
end




"""
For the BA case of the G|v> calculation:
# 1. embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], cellsA[1]+cellsB[1], cellsA[2]+cellsB[2], cellsA[3]+cellsB[3])
    # 2. fft of the embedded v and of g 
    # creation of the fft plan
    x=rand(ComplexF64, length(v_embedded), 1) 
    plan = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf) 
    fft_v_embedded = plan*v_embedded
    fft_g = plan*g
    # 3. element-wise multiplication of g and v
    mult = fft_g .* fft_v_embedded
    # 4. inverse fft of the multiplication of g and v 
    # creation of the inverse fft plan
    p_inv = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
    fft_inv = p_inv*mult
    # 5. project the fft inverse 
    projection = proj(fft_inv, cellsA[1]+cellsB[1], cellsA[2]+cellsB[2], cellsA[3]+cellsB[3], cellsB[1], cellsB[2], cellsB[3])
    return projection 
"""