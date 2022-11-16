import G_v_product
module Gdag_v_product
export Embedding, Projection, FFTW, Gdag_v
using Embedding, Projection, FFTW 
# function for Gdag|v>, where |v> is a vector 
# when working with gdag, the order of the fft 
# and the inverse fft are swapped. 
function Gdag_v(g, v, cellsA) 
    # calculate gdag 
    # the code below has the same effect as calculating gddag, 
    # which would normally be done like: gdag = conj.(transpose(g)) 
    # embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], 2*cellsA[1], 2*cellsA[2], 2*cellsA[3])
    # inverse fft of the embedded v and of gdag
    # calculate the direct fft of g_dag 
    fft_g = fft_plan*g 
    # calculate the inverse fft of the embedded vector
    fftinv_v_embedded = inv_fft_plan*v_embedded
    # 3. element-wise multiplication of g and v
    mult = fft_g .* fftinv_v_embedded
    # 4. fft of the multiplication of g and v 
    # fft of the product
    fft_prod = fft_plan*mult
    # 5. project the fft inverse 
    projection = proj(fft_prod, 2*cellsA[1], 2*cellsA[2], 2*cellsA[3], cellsA[1], cellsA[2], cellsA[3])
    return projection 
end
end