using Distributed
@everywhere using Printf, Base.Threads, LinearAlgebra.BLAS, MaxGParallelUtilities, MaxGStructs, 
MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, MaxGCUDA, MaxGOpr, product, Random 
# ,bfgs_power_iteration_asym_only, dual_asym_only, gmres

# , dual_asym_only, bfgs_power_iteration_asym_only
# G_v_product,

# The following latex files explains some of the different functions of the program
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

# Setup for the creation of the total Green function
# Start 
# Define test volume, all lengths are defined relative to the wavelength. 
# Number of cells in the volume. 
cellsA = [8, 8, 8]
cellsB = [1, 1, 1]

print("Hello 1")

# Edge lengths of a cell relative to the wavelength. 
scaleA = (0.02, 0.02, 0.02)
scaleB = (0.02, 0.02, 0.02)
# Center position of the volume. 
coordA = (0.0, 0.0, 0.0)
coordB = (-0.3, 0.0, 0.0)
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
greenCircAB = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
	cellsA[2], cellsB[3] + cellsA[3]) # The order of G_{1,2} -> 1: target, 2: source
# End 

# CPU computation of Green function
# Start 
# The first index is the target and the second is the source
# For the AA case 
genGreenSlf!(greenCircAA, volA, assemblyInfo)
# For the AB case 
genGreenExt!(greenCircBA, volA, volB, assemblyInfo)
# End 

# Nic's optimisation code: 

# Double check this section with Sean -> Already did and he said all is good. 
# The code below is for the creation of the incident electric field 
# e_i = (G_{AB})(j_i^A) where j_i^A has dimensions of (cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
print("Hello 2")
ji_A = Array{ComplexF64}(undef, cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
rand!(ji_A) # Not sure what to put as ji_A, so let's just put a random vector for now 
# To do the Gv_AB product, we need to reshalpe the input vector ji_A
# into the dimensions of currSrcAB = Array{ComplexF64}(undef, cellsB[1], cellsB[2], cellsB[3], 3).
# To use the output of the Gv_AB product in the rest of the code, we need to reshape it back into
# a vector that has dimensions (cellsA[1]*cellsA[2]*cellsA[3]*3, 1)
ei = reshape(Gv_AB(greenCircAB, cellsA, cellsB, reshape(ji_A,(cellsB[1], cellsB[2], cellsB[3], 3))),(cellsA[1]*cellsA[2]*cellsA[3]*3, 1))
print(ei)
# End 

# Let's define some values used throughout the program.
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



# Code used for tests 
# Start 
# # srcSize = (cellsA[1], cellsA[2], cellsA[3])
# # # total cells 
# # totCells = prod(srcSize)

# rand!(currSrc)
# greenAct!()
# print(currTrg)
# End 

# Generate a test vect 
# testvect = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# rand!(testvect)

# # Test that does does an offset to the testvect
# offset = 5
# new = testvect .+ offset 
# print("testvect ", testvect, "\n")
# print("new ", new, "\n")

# # The two lines below are used for the vdag_GAsym_v calculation
# srcSize = (cellsA[1], cellsA[2], cellsA[3])
# totCells = prod(srcSize)

# print("Gv(greenCircAA, cellsA, testvect) ", Gv(greenCircAA, cellsA, testvect), "\n")
# print("GAdjv(greenCircAA, cellsA, testvect) ", GAdjv(greenCircAA, cellsA, testvect), "\n")
# Good code 
# Start
# GAsym_v_prod = (Gv(greenCircAA, cellsA, testvect)-GAdjv(greenCircAA, cellsA, testvect))/2.0im
# vdag_GAsym_v = dotc(3*totCells, testvect, 1, GAsym_v_prod, 1)
# print("vdag_GAsym_v ", vdag_GAsym_v, "\n")
# End

# Code below is to determine the offset needed 
# Start 
# i = 0 # This is needed if a while loop is used instead of a for loop
# There seems to be no need for an offset since the complex parts are negligeable and 
# I seem to always get one when I run the first check code below, so always positive values.

# offset1 = 0
# testvect = Array{ComplexF64}(undef, cellsA[1], cellsA[2], cellsA[3], 3)
# for i = 1:100
# 	print("Generated test vector \n")
# 	rand!(testvect)
# 	offset1 = 0.0 + 1e-4im
# 	offset = 0.0 + 0.0im
# 	first_check = false
# 	second_check = true
# 	while first_check != true
# 		GAsym_v_prod = (Gv(greenCircAA, cellsA, testvect, offset1)-GAdjv(greenCircAA, cellsA, testvect,offset1))/2.0im
# 		vdag_GAsym_v = dotc(3*totCells, testvect, 1, GAsym_v_prod, 1)
# 		vdag_v = dotc(3*totCells, testvect, 1, testvect, 1)
# 		result = vdag_GAsym_v/vdag_v
# 		offset += abs(result)*1im 
# 		print("result ", result, "\n")
# 		if real.(result) < 0 
# 			offset1 .+= offset
# 		else
# 			first_check = true
# 		end
# 	end
	# old_offset = offset1
	# new_offset = 0.0 + 0.0im
	# while second_check == true
	# 	testvect .-= offset1/2
	# 	GAsym_v_prod = (Gv(greenCircAA, cellsA, testvect)-GAdj_v_prod(greenCircAA, cellsA, testvect))/2.0im
	# 	vdag_GAsym_v = dotc(testvect, GAsym_v_prod)
	# 	vdag_v = dotc(testvect, testvect)
	# 	result = vdag_GAsym_v/vdag
	# 	if result < 0 
	# 		testvect = testvect + offset1 
	# 		second_check = false # not really needed since we will just return the last offset that worked
	# 		return 
	# 	else
	# 	end
	# end 
	# global i += 1 
# end

# print("offset1 ", offset1, "\n")
# End 

# This is the code for the main function call using bfgs with the power iteration
# method to solve for the Lagrange multiplier and gmres to solve for |T>.
# Start 
# bfgs = BFGS_fakeS_with_restart_pi(greenCircAA,l,dual,P,chi_inv_coeff,ei,e_vac,
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
# End 