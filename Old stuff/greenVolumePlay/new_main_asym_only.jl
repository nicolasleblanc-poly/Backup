using Distributed
@everywhere using Printf, Base.Threads, LinearAlgebra.BLAS, MaxGParallelUtilities, MaxGStructs, 
MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA,  MaxGOpr, product, Random

# , dual_asym_only, bfgs_power_iteration_asym_only
# G_v_product,

# The following latex files explains the different functions of the program
# https://www.overleaf.com/read/yrdmwzjhqqqs

# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
threads = nthreads()
BLAS.set_num_threads(threads)
blasThreads = BLAS.get_num_threads()
println("MaxG test initialized with ", nthreads(), 
	" Julia threads and $blasThreads BLAS threads.")
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [16, 16, 16]
cellsB = [1, 1, 1]
# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.02, 0.02, 0.02)
scaleB = (0.02, 0.02, 0.02)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (-0.3, 0.3, 0.0)
# Create MaxG volumes.
volA = genMaxGVol(MaxGDom(cellsA, scaleA, coordA))
volB = genMaxGVol(MaxGDom(cellsB, scaleB, coordB))
# Information for Green function construction. 
# Complex frequency ratio. 
freqPhase = 1.0 + im * 0.0
# Gauss-Legendre approximation orders. 
ordGLIntFar = 2
ordGLIntMed = 8
ordGLIntNear = 16
# Cross over points for Gauss-Legendre approximation.
crossMedFar = 16
crossNearMed = 8
assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
	ordGLIntNear, crossMedFar, crossNearMed)
# Pre-allocate memory for circulant green function vector. 
# Let's say we are only considering the AA case for simplicity
greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
	2 * cellsA[3])
# greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])

# CPU computation of Green function
genGreenSlf!(greenCircAA, volA, assemblyInfo)
# genGreenExt!(greenCircBA, volB, volA, assemblyInfo)


# Sean's eigenvalue test code:
### Eigenvalue test
# include("eigTest.jl")

# Analytic test
# # separation vector
# s = 2 * pi * sqrt((coordB[1])^2 + coordB[2]^2 + coordB[3]^2)
# sh = (2 * pi) .* (coordB[1], coordB[2], coordB[3]) ./ s 
# # operators
# id = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
# sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); 
# (sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]); 
# (sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]
# # analytic green function
# gAna = (exp(im * s) / (4 * pi * s)) .* (((1 + (im * s - 1) / s^2) .* id) .- 
# ((1 + 3 * (im * s - 1) / s^2) .* sHs))

# conFac = 4.0314419358490252e-3 - 4.907076775369318e-12im

# ratio = (gAna ./ greenCircBA[:, :, 1, 1, 1]) .* (scale^3 / conFac)

return nothing;
# cudaLibPtr = libInitCu!()
# devNum = devCount(cudaLibPtr)
# print("Found ", devNum, " devices.\n")
# libFinlCu!(cudaLibPtr)



# Nic's optimisation code: 


# Double check this section with Sean 
# separation vector
# s = 0
# sh = ones(ComplexF64, 3, 1) #double check if this is the right size of the initial ji
# # it is better to pre-allocate memory than to append to an empty vector when you now the 
# # size of what you are working with
# # Pre-allocate memory for initial electric field in the different directions of space. 
# dim = cellsA[1]*cellsA[2]*cellsA[3]
# ei = Array{ComplexF64}(undef,3*dim,1) # things are stored in ei like: [ei_x, ei_y, ei_z]
# iteration = 1
# for cellZ = 1:cellsA[3]
#     for cellY = 1:cellsA[2]
#         for cellX = 1:cellsA[1]
#             global s = 2 * pi * sqrt((scaleA[1] * cellX-seperation)^2 + (scaleA[2] * cellY)^2 + (scaleA[3] * cellZ)^2)
#             global sh = (2 * pi) .* ((scaleA[1] * cellX-seperation), scaleA[2]* cellY, scaleA[3] * cellZ) ./ s
#             # let's consider a dipole oriented in the z-direction
#             id = [0.0 0.0 1.0]
#             sHs = [(sh[1] * sh[3]);  (sh[2] * sh[3]); (sh[3] * sh[3])] #xz, yz and zz
#             # analytic green function
#             gAna = (exp(im * s) / (4 * pi * s)) .* (((1 + (im * s - 1) / s^2) .* id) .- ((1 + 3 * (im * s - 1) / s^2) .* sHs))
#             ei[iteration] = gAna[1] # ei_x 
#             ei[iteration+dim] = gAna[2] # ei_y 
#             ei[iteration+2*dim] = gAna[3] # ei_z
#             # ei is stored like: [ei_x, ei_y, ei_z]
#             global iteration += 1
#         end
#     end    
# end
# Double check all of this with Sean for the seperation vector

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
# # xx 
# g_xx = greenCircAA[1,1,:,:,:] 
# # xy
# g_xy = greenCircAA[1,2,:,:,:]
# # xz 
# g_xz = greenCircAA[1,3,:,:,:]
# # yx
# g_yx = greenCircAA[2,1,:,:,:]
# # yy 
# g_yy = greenCircAA[2,2,:,:,:]
# # yz 
# g_yz = greenCircAA[2,3,:,:,:]
# # zx 
# g_zx = greenCircAA[3,1,:,:,:]
# # zy
# g_zy = greenCircAA[3,2,:,:,:]
# # zz 
# g_zz = greenCircAA[3,3,:,:,:]

e_vac = 0 # Check how to find its value with Sean

# direct fft of the different components of the Green function
# # xx 
# g_dff_xx = greenAct![1,1,:,:,:] 
# # xy
# g_dff_xy = greenAct![1,2,:,:,:]
# # xz 
# g_dff_xz = greenAct![1,3,:,:,:]
# # yx
# g_dff_yx = greenAct![2,1,:,:,:]
# # yy 
# g_dff_yy = greenAct![2,2,:,:,:]
# # yz 
# g_dff_yz = greenAct![2,3,:,:,:]
# # zx 
# g_dff_zx = greenAct![3,1,:,:,:]
# # zy
# g_dff_zy = greenAct![3,2,:,:,:]
# # zz 
# g_dff_zz = greenAct![3,3,:,:,:]

# # inverse fft fft of the different components of the Green function  
# # xx 
# g_ifft_xx = greenAdjAct![1,1,:,:,:] 
# # xy
# g_ifft_xy = greenAdjAct![1,2,:,:,:]
# # xz 
# g_ifft_xz = greenAdjAct![1,3,:,:,:]
# # yx
# g_ifft_yx = greenAdjAct![2,1,:,:,:]
# # yy 
# g_ifft_yy = greenAdjAct![2,2,:,:,:]
# # yz 
# g_ifft_yz = greenAdjAct![2,3,:,:,:]
# # zx 
# g_ifft_zx = greenAdjAct![3,1,:,:,:]
# # zy
# g_ifft_zy = greenAdjAct![3,2,:,:,:]
# # zz 
# g_ifft_zz = greenAdjAct![3,3,:,:,:]


# Positive definite test
# We want (vdag G^A v)/(vdag v) to be larger or equal to 0. This would ensure that 
# all the eigenvalues are positive and therefore our operator is positive definite 
# The size of the rdn vect with depend on if we use a part of G or all of G
# Source current memory
test_current = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
testvect = rand!(test_current)
GAsym_v_prod = (G_v_prod(greenCircAA, cellsA, testvect)-GAdj_v_prod(greenCircAA, cellsA, testvect))/2.0im
vdag_GAsym_v = dotc(testvect, GAsym_v_prod)
vdag_v = dotc(testvect, testvect)
result = vdag_GAsym_v/vdag
print("result ", result, "\n")

# i = 0 
# while i < 100
# 	testvect = rand!(test_current)
# 	offset1 = 0.0 + 0.0im
# 	first_check = false
# 	second_check = true
# 	while first_check != true
# 		GAsym_v_prod = (G_v_prod(cellsA, testvect)-GAdj_v_prod(cellsA, testvect))/2.0im
# 		vdag_GAsym_v = dotc(testvect, GAsym_v_prod)
# 		vdag_v = dotc(testvect, testvect)
# 		result = vdag_GAsym_v/vdag
# 		offset1 += abs(result)*1im 
# 		if result < 0 
# 			testvect += offset1 
# 		else
# 			first_check = true
# 		end
# 	end
# 	old_offset = offset1
# 	new_offset = 0.0 + 0.0im
# 	while second_check == true
# 		testvect -= offset1/2
# 		GAsym_v_prod = (G_v_prod(cellsA, testvect)-GAdj_v_prod(cellsA, testvect))/2.0im
# 		vdag_GAsym_v = dotc(testvect, GAsym_v_prod)
# 		vdag_v = dotc(testvect, testvect)
# 		result = vdag_GAsym_v/vdag
# 		if result < 0 
# 			testvect = testvect + offset1 
# 			second_check = false # not really needed since we will just return the last offset that worked
# 			return 
# 		else
# 			new_
# 		end
# 	end 
# 	global i += 1 
# end

# main function call
# bfgs = BFGS_fakeS_with_restart_pi(l,dual,P,ei,e_vac,fft_plan_x,fft_plan_y,fft_plan_z,inv_fft_plan_x,inv_fft_plan_y,inv_fft_plan_z,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,
# g_dff_xx, g_dff_xy, g_dff_xz, g_dff_yx, g_dff_yy, g_dff_yz, g_dff_zx, g_dff_zy, g_dff_zz,
# g_iff_xx, g_iff_xy, g_iff_xz, g_iff_yx, g_iff_yy, g_iff_yz, g_iff_zx, g_iff_zy, g_iff_zz,
# cellsA,validityfunc,power_iteration_second_evaluation)
# # the BFGS_fakeS_with_restart_pi function can be found in the bfgs_power_iteration_asym_only file
# dof = bfgs[1]
# grad = bfgs[2]
# dualval = bfgs[3]
# objval = bfgs[4]
# print("dof ", dof, "\n")
# print("grad ", grad, "\n")
# print("dualval ", dualval, "\n")
# print("objval ", objval, "\n")
