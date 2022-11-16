using Distributed
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, dual_asym_only, bfgs_power_iteration_asym_only, FFTW, G_v_product
# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
"""
Major changes made on August 11th 2022
- Added a minus sign in front of the second part of the b calculation (aka (l[1]/(2im))*P*ei )
- Added indexing for each direction in space when doing the G|v> and Gdag|v> products 
    example for the z-direction and gdag: 
    -> x part: o_zx = G_v(g_zx, v[1:8], fft_plan,inv_fft_plan,cellsA)
    -> y part: o_zy = G_v(g_zy, v[9:16], fft_plan,inv_fft_plan,cellsA)
    -> z part: o_zz = G_v(g_zz, v[17:24], fft_plan,inv_fft_plan,cellsA)
"""

global Z = 1
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
# it is better to pre-allocate memory than to append to an empty vector when you now the 
# size of what you are working with
# Pre-allocate memory for initial electric field in the different directions of space. 
dim = cellsA[1]*cellsA[2]*cellsA[3]
ei = Array{ComplexF64}(undef,3*dim,1) # things are stored in ei like: [ei_x, ei_y, ei_z]
# ei_x = Array{ComplexF64}(undef,dim,1)
# ei_y = Array{ComplexF64}(undef,dim,1)
# ei_z = Array{ComplexF64}(undef,dim,1)
iteration = 1
for cellZ = 1:cellsA[3]
    for cellY = 1:cellsA[2]
        for cellX = 1:cellsA[1]
            global s = 2 * pi * sqrt((scale * cellX-seperation)^2 + (scale * cellY)^2 + (scale * cellZ)^2)
            global sh = (2 * pi) .* ((scale * cellX-seperation), scale* cellY, scale * cellZ) ./ s
            # the two lines below are errors that I (Nic) made 
            # global s = 2 * pi * sqrt((cellX-seperation)^2 + cellY^2 + cellZ^2)
            # global sh = (2 * pi) .* ((cellX-seperation), cellX, cellZ) ./ s 
            # operators
            # let's consider a dipole oriented in the z-direction
            id = [0.0 0.0 1.0]
            sHs = [(sh[1] * sh[3]);  (sh[2] * sh[3]); (sh[3] * sh[3])] #xz, yz and zz
            # analytic green function
            gAna = (exp(im * s) / (4 * pi * s)) .* (((1 + (im * s - 1) / s^2) .* id) .- ((1 + 3 * (im * s - 1) / s^2) .* sHs))
            ei[iteration] = gAna[1] # ei_x 
            ei[iteration+dim] = gAna[2] # ei_y 
            ei[iteration+2*dim] = gAna[3] # ei_z
            # ei is stored like: [ei_x, ei_y, ei_z]
            global iteration += 1
            # another way of doing it
            # ei_x[iteration] = gAna[1]
            # ei_y[iteration] = gAna[2]
            # ei_z[iteration] = gAna[3]
            # the code below is a bad practice when you know the size of what you are working with
            # append!(ei_x,gAna[1])
            # append!(ei_y,gAna[2])
            # append!(ei_z,gAna[3])
        end
    end    
end

# print("length(ei_x) ", length(ei_x), "\n")
# print("length(ei_y) ", length(ei_y), "\n")
# print("length(ei_z) ", length(ei_z), "\n")


# ei = [ei_x, ei_y, ei_z] # total electric field

# Notes to improve programming (Sean's pro tips):
# put fft in main fft and inverse fft to not have to recalculate or use a global variable instead of calculating a new plan everytime
# don't use append if you know how long something is
# reshape is better than vect function you wrote

# let's define some values
# chi coefficient
chi_coeff = 3.0 + 0.01im
# inverse chi coefficient
chi_inv_coeff = 1/chi_coeff 
chi_inv_coeff_dag = conj(chi_inv_coeff)
# define the projection operators
P = I # this is the real version of the identity matrix since we are considering 
# the symmetric and ansymmetric parts of some later calculations. 
# If we were only considering the symmetric parts of some latter calculations,
# we would need to use the imaginary version of the identity matrix. 
# Pdag = conj.(transpose(P)) 
# we could do the code above for a P other than the identity matrix
Pdag = P
# let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
# b = bv(ei, l,P)
l = [0.5] # initial Lagrange multipliers

# Green functions
# xx 
gf_xx = greenCircAA[1,1,:,:,:]
# xy
gf_xy = greenCircAA[1,2,:,:,:]
# xz 
gf_xz = greenCircAA[1,3,:,:,:]
# yx
gf_yx = greenCircAA[2,1,:,:,:]
# yy 
gf_yy = greenCircAA[2,2,:,:,:]
# yz 
gf_yz = greenCircAA[2,3,:,:,:]
# zx 
gf_zx = greenCircAA[3,1,:,:,:]
# zy
gf_zy = greenCircAA[3,2,:,:,:]
# zz 
gf_zz = greenCircAA[3,3,:,:,:]
# print("size(g_zz) ", gf_zz, "\n")
# let's turn the Green functions into vectors using linear indexing
# it's better to use reshape than the vect function I wrote since it's more efficient
# xx 
g_xx = reshape(gf_xx, (length(gf_xx), 1))
# xy
g_xy = reshape(gf_xy, (length(gf_xy), 1))
# xz 
g_xz = reshape(gf_xx, (length(gf_xz), 1))
# yx
g_yx = reshape(gf_xx, (length(gf_yx), 1))
# yy 
g_yy = reshape(gf_xx, (length(gf_yy), 1))
# yz 
g_yz = reshape(gf_xx, (length(gf_yz), 1))
# zx 
g_zx = reshape(gf_xx, (length(gf_zx), 1))
# zy
g_zy = reshape(gf_xx, (length(gf_zy), 1))
# zz 
g_zz = reshape(gf_xx, (length(gf_zz), 1))

e_vac = 0 #check it's value with Sean

# creation of direct and inverse fft plan
# they are done here since they are used multiple times and creating 
# them as global variables allows to avoid generating them everytime they are used
x=rand(ComplexF64, 8*dim, 1) # *3, since we are considering all 3 directions at the same time
# calculation of the direct fft plan 
fft_plan = plan_fft(x; flags=FFTW.ESTIMATE, timelimit=Inf) 
# calculation of the inverse fft plan
inv_fft_plan = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)


# testvect = rand(ComplexF64,24,1)
# print("testvect ", testvect, "\n")

# # whole = output(l,testvect,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
# # print("whole ", whole, "\n")
# asym_testvect = output(l,testvect,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
# testvect_asym_testvect = conj.(transpose(testvect))*asym_testvect
# print("testvect_asym_testvect ", testvect_asym_testvect*l[1], "\n")

# main function call
bfgs = BFGS_fakeS_with_restart_pi(l,dual,P,ei,e_vac,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,validityfunc,power_iteration_second_evaluation)
dof = bfgs[1]
grad = bfgs[2]
dualval = bfgs[3]
objval = bfgs[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")