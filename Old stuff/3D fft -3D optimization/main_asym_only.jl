using Distributed 
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, dual_asym_only, bfgs_power_iteration_asym_only, FFTW, G_v_product
# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 

# The following latex files explains the different functions of the program
# https://www.overleaf.com/read/yrdmwzjhqqqs

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
out_fft = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
2 * cellsA[3])

# separation vector
s = 0
sh = ones(ComplexF64, 3, 1) #double check if this is the right size of the initial ji
# it is better to pre-allocate memory than to append to an empty vector when you now the 
# size of what you are working with
# Pre-allocate memory for initial electric field in the different directions of space. 
dim = cellsA[1]*cellsA[2]*cellsA[3]
ei = Array{ComplexF64}(undef,3*dim,1) # things are stored in ei like: [ei_x, ei_y, ei_z]
iteration = 1
for cellZ = 1:cellsA[3]
    for cellY = 1:cellsA[2]
        for cellX = 1:cellsA[1]
            global s = 2 * pi * sqrt((scale * cellX-seperation)^2 + (scale * cellY)^2 + (scale * cellZ)^2)
            global sh = (2 * pi) .* ((scale * cellX-seperation), scale* cellY, scale * cellZ) ./ s
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
        end
    end    
end
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

# array Green functions 
# the values below are technically gf_xx or other but for simplicity I'm calling them 
# g_xx or other since I don't want to have to change all the g_xx's everywhere in my code
# xx 
g_xx = greenCircAA[1,1,:,:,:] 
# xy
g_xy = greenCircAA[1,2,:,:,:]
# xz 
g_xz = greenCircAA[1,3,:,:,:]
# yx
g_yx = greenCircAA[2,1,:,:,:]
# yy 
g_yy = greenCircAA[2,2,:,:,:]
# yz 
g_yz = greenCircAA[2,3,:,:,:]
# zx 
g_zx = greenCircAA[3,1,:,:,:]
# zy
g_zy = greenCircAA[3,2,:,:,:]
# zz 
g_zz = greenCircAA[3,3,:,:,:]
# print("size(g_zz) ", gf_zz, "\n")
# let's turn the Green functions into vectors using linear indexing
# it's better to use reshape than the vect function I wrote since it's more efficient
# xx 
# g_xx = reshape(gf_xx, (length(gf_xx), 1))
# # xy
# g_xy = reshape(gf_xy, (length(gf_xy), 1))
# # xz 
# g_xz = reshape(gf_xx, (length(gf_xz), 1))
# # yx
# g_yx = reshape(gf_xx, (length(gf_yx), 1))
# # yy 
# g_yy = reshape(gf_xx, (length(gf_yy), 1))
# # yz 
# g_yz = reshape(gf_xx, (length(gf_yz), 1))
# # zx 
# g_zx = reshape(gf_xx, (length(gf_zx), 1))
# # zy
# g_zy = reshape(gf_xx, (length(gf_zy), 1))
# # zz 
# g_zz = reshape(gf_xx, (length(gf_zz), 1))

e_vac = 0 #check it's value with Sean

# creation of direct and inverse fft plan for input which are arrays
# they are done here since they are used multiple times and creating 
# them as global variables allows to avoid generating them everytime they are used
# calculation of the direct x fft plan 
fft_plan_x = plan_fft(g_xx, 1) 
# calculation of the direct y fft plan 
fft_plan_y = plan_fft(g_xx, 2)
# calculation of the direct z fft plan 
fft_plan_z = plan_fft(g_xx, 3)
# calculation of the inverse x fft plan
inv_fft_plan_x = plan_ifft(g_xx, 1)
# calculation of the inverse x fft plan
inv_fft_plan_y = plan_ifft(g_xx, 2)
# calculation of the inverse x fft plan
inv_fft_plan_z = plan_ifft(g_xx, 3)
# the plan_fft and plan_ifft functions are part of the FFTW package
# this thread exaplains how to take an fft along a direction (x, y or z): https://discourse.julialang.org/t/better-documentation-of-dims-for-julia-fft-function/9016/2
# x is down, y is towards the right and z is out of the page

# code to test to see if we get gf_xx after doing the direct fft and then the 
# inverse fft of the result of the direct fft
# print("gf_xx ", g_xx, "\n")
# test_1 = fft_plan_x*g_xx
# test_2 = fft_plan_y*test_1
# test_3 = fft_plan_z*(fft_plan_y*(fft_plan_x*g_xx)) # how to do x, then y and then z 
# print("test_3 ", test_3, "\n")
# test_x = fft_plan_z*fft_plan_y*fft_plan_x*g_xx
# print("test_x ", test_x, "\n")
# test_y = fft_plan_y*g_xx
# print("test_y ", test_y, "\n")
# test_z = fft_plan_z*g_xx
# print("test_z ", test_z, "\n")
# inv_test_x = inv_fft_plan_x*test_x
# print("inv_test_x ", inv_test_x, "\n")
# inv_test_y = inv_fft_plan_y*test_y
# print("inv_test_y ", inv_test_y, "\n")
# inv_test_z = inv_fft_plan_z*test_z
# print("inv_test_z ", inv_test_z, "\n")

# test code 
# testvect = rand(ComplexF64,24,1)
# print("testvect ", testvect, "\n")

# # # whole = output(l,testvect,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
# # # print("whole ", whole, "\n")
# asym_testvect = output(l,testvect,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
# testvect_asym_testvect = conj.(transpose(testvect))*asym_testvect
# print("testvect_asym_testvect ", testvect_asym_testvect*l[1], "\n")
# # end of test code

# main function call
bfgs = BFGS_fakeS_with_restart_pi(l,dual,P,ei,e_vac,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA,validityfunc,power_iteration_second_evaluation)
the BFGS_fakeS_with_restart_pi function can be found in the bfgs_power_iteration_asym_only file
dof = bfgs[1]
grad = bfgs[2]
dualval = bfgs[3]
objval = bfgs[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")


"""
Values of last run:
stepsize alphaopt is 4.176729405444104e-16
delta [-4.176729405444104e-16;;]
T_asym_T ComplexF64[-2.587582128148787e-13 + 0.0im;;]
delta[-4.176729405444104e-16;;]
dualval -3.325437098807045e-13
objval -3.828578068169107e-13
eqcstval 5.031409693620619e-14
at iteration # 20 the dual, objective, eqconstraint value is -3.325437098807045e-13  -3.828578068169107e-13  5.031409693620619e-14
normgrad is 1.437545626748829e-14
prev_dualval is -3.3254370988070663e-13
dof [3.4999999999998037;;]
grad [1.437545626748829e-14;;]
dualval -3.325437098807045e-13
objval -3.828578068169107e-13
"""

"""
ji_embedded output 
[0.09221023336821073 - 0.17153401901769097im
0.061650266485992665 - 0.30407498537292327im
0.0 + 0.0im
0.0 + 0.0im
-0.03973163007238337 - 0.1766381834485968im
-0.025688652371394387 + 0.02949317956327182im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
-0.11237957634803278 + 0.0018662723479086845im
-0.06267452442715575 - 0.10440453911431131im
0.0 + 0.0im
0.0 + 0.0im
-0.07783427775875325 - 0.02900849181702065im
-0.023845651350592502 + 0.16167398536768585im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im 
0.0 + 0.0im
0.0 + 0.0im
0.0 + 0.0im 
0.0 + 0.0im 
0.0 + 0.0im 
0.0 + 0.0im 
0.0 + 0.0im]
"""