using Distributed
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, Embedding, Projection, FFTW
# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
threads = nthreads()
BLAS.set_num_threads(threads)
blasThreads = BLAS.get_num_threads()
println("MaxG test initialized with ", nthreads(), " Julia threads and $blasThreads BLAS threads.")
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [2, 2, 2]
cellsB = [1, 1, 1]
# let's definite the initial Lagrange multipliers
l = zeros(Float64, 2, 1)
l[1] = 1 # asymmetric part
# Size length of a cell relative to the wavelength. 
scale = 0.0001
# Center position of the volume. 
seperation = 2
coordA = (0.0, 0.0, 0.0)
coordB = (-seperation, 0.0, 0.0)
# Create MaxG volumes.
volA = genMaxGVol(MaxGDom(cellsA, scale, coordA))
volB = genMaxGVol(MaxGDom(cellsB, scale, coordB))
# Information for Green function construction. 
# Complex frequency ratio. 
freqPhase = 1.0 + im * 0.0
# Gauss-Legendre approximation orders. 
ordGLIntFar = 2
ordGLIntMed = 4
ordGLIntNear = 16
# Cross over points for Gauss-Legendre approximation.
crossMedFar = 16
crossNearMed = 8
assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, ordGLIntNear, crossMedFar, crossNearMed)
# Pre-allocate memory for circulant green function vector. 
greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
    2 * cellsA[3])
# CPU computation of Green function
genGreenSlf!(greenCircAA, volA, assemblyInfo)
# separation vector
s = 0
sh = ones(ComplexF64, 3, 1) #double check if this is the right size of the initial ji
for cellZ = 1:cellsB[3]
    for cellY = 1:cellsB[2]
        for cellX = 1:cellsB[1]
            global s = 2 * pi * sqrt((cellX-seperation)^2 + cellY^2 + cellZ^2)
            global sh = (2 * pi) .* ((cellX-seperation), cellX, cellZ) ./ s 
        end
    end    
end
# operators
id = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]
sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); (sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]); (sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]
sHs_z = sHs*id
# initial electric fields 
ei_x = sHs_z[7]
ei_y = sHs_z[8]
ei_z = sHs_z[9]
ei = zeros(ComplexF64, 3, 1)
ei[1] = ei_x 
ei[2] = ei_y 
ei[3] = ei_z
# function for G|v>, where |v> is a vector  
function G_v(g, v) 
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
end
# creation of b 
function bv(ei, l) 
    P = I
    return -ei/(2im) + (l[1]/2)*P*ei + (l[2]/(2im))*P*ei
end
# let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
b = bv(ei, l)
# Green functions
# xx 
gf_xx = greenCircBA[1,1,:,:,:]
# xy
gf_xy = greenCircBA[1,2,:,:,:]
# xz 
gf_xz = greenCircBA[1,3,:,:,:]
# yx
gf_yx = greenCircBA[2,1,:,:,:]
# yy 
gf_yy = greenCircBA[2,2,:,:,:]
# yz 
gf_yz = greenCircBA[2,3,:,:,:]
# zx 
gf_zx = greenCircBA[3,1,:,:,:]
# zy
gf_zy = greenCircBA[3,2,:,:,:]
# zz 
gf_zz = greenCircBA[3,3,:,:,:]
# turn the Green functions into vectors using linear indexing
function gvec(g)
    index = 1
    g_vector = zeros(ComplexF64, 8, 1)
    for element in g
        if index < 8
            g_vector[index] = element
            global index += 1
        end
    end
    return g_vector
end
# xx 
g_xx = gvec(gf_xx)
# xy
g_xy = gvec(gf_xy)
# xz 
g_xz = gvec(gf_xz)
# yx
g_yx = gvec(gf_yx)
# yy 
g_yy = gvec(gf_yy)
# yz 
g_yz = gvec(gf_yz)
# zx 
g_zx = gvec(gf_zx)
# zy
g_zy = gvec(gf_zy)
# zz 
g_zz = gvec(gf_zz)
# let's get the output (Green function?) in the x, in the y and in the z by calculating G|v>, where |v> is b 
# xx ouput 
o_xx = G_v(g_xx, b)
# xy ouput
o_xy = G_v(g_xy, b)
# xz ouput
o_xz = G_v(g_xz, b)
# yx ouput
o_yx = G_v(g_yx, b)
# yy ouput
o_yy = G_v(g_yy, b)
# yz ouput
o_yz = G_v(g_yz, b)
# zx ouput
o_zx = G_v(g_zx, b)
# zy ouput
o_zy = G_v(g_zy, b)
# zz ouput
o_zz = G_v(g_zz, b)
# let's get the output in each direction of space (x, y and z)
o_x = o_xx + o_xy + o_xz 
o_y = o_yx + o_yy + o_yz 
o_z = o_zx + o_zy + o_zz 
# we use these outputs as the G in objective and constraints of Sean's articles.