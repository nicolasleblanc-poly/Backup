module G_v_product
export G_v, output
using Embedding, Projection, FFTW
function output(g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,v,cellsA)
    o_xx = G_v(g_xx, v, cellsA)
    # xy ouput
    o_xy = G_v(g_xy, v, cellsA)
    # xz ouput
    o_xz = G_v(g_xz, v, cellsA)
    # x total output
    o_x = o_xx + o_xy + o_xz 
    # xx ouput 
    o_xx_dag = G_v(g_xx_dag, v, cellsA)
    # xy ouput
    o_xy_dag = G_v(g_xy_dag, v, cellsA)
    # xz ouput
    o_xz_dag = G_v(g_xz_dag, v, cellsA)
    # x total dag output
    o_x_dag = o_xx_dag + o_xy_dag + o_xz_dag 
    return o_x, o_x_dag
end



# function for G|v>, where |v> is a vector 
# code for the AA case 
function G_v(g, v, cellsA) # add cellsB for the BA case 
    # 1. embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
    # 2. fft of the embedded v and of g 
    # creation of the fft plan
    x=rand(ComplexF64, length(v_embedded), 1) 
    plan = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf) 
    fft_v_embedded = plan*v_embedded
    # print("size(v_embedded) ", size(v_embedded), "\n")
    # print("size(g) ", size(g), "\n")
    fft_g = plan*g
    # 3. element-wise multiplication of g and v
    mult = fft_g .* fft_v_embedded
    # 4. inverse fft of the multiplication of g and v 
    # creation of the inverse fft plan
    p_inv = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
    fft_inv = p_inv*mult
    # 5. project the fft inverse 
    projection = proj(fft_inv, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
    return projection 
end
end

"""
For the BA case:
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