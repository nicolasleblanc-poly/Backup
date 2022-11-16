module Gdag_v_product
export Gdag_v
# function for Gdag|v>, where |v> is a vector 
# when working with gdag, the order of the fft 
# and the inverse fft are swapped. 
function Gdag_v(g, v) 
    # calculate gdag 
    gdag = conj.(transpose(g)) 
    # embed v 
    v_embedded = embedVec(v, cellsA[1], cellsA[2], cellsA[3], cellsA[1]+cellsB[1], cellsA[2]+cellsB[2], cellsA[3]+cellsB[3])
    # inverse fft of the embedded v and of gdag
    # creation of the direct fft plan
    x=rand(ComplexF64, length(v_embedded), 1) 
    plan = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf) 
    # creation of the inverse fft plan
    p_inv = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
    fftinv_v_embedded = p_inv*v_embedded
    fft_gdag = p*gdag
    # 3. element-wise multiplication of g and v
    mult = fft_gdag .* fftinv_v_embedded
    # 4. fft of the multiplication of g and v 
    # fft of the product
    fft_prod = plan*mult
    # 5. project the fft inverse 
    projection = proj(fft_prod, cellsA[1]+cellsB[1], cellsA[2]+cellsB[2], cellsA[3]+cellsB[3], cellsB[1], cellsB[2], cellsB[3])
    return projection 
end
end