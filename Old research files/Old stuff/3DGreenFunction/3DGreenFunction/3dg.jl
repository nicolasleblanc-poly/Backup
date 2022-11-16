using Distributed
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, 
MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, Embedding, Projection, FFTW
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
# cellsA = [4, 4, 4]
# cellsB = [4, 4, 4]
# # Size length of a cell relative to the wavelength. 
# scale = 0.01
# # Center position of the volume. 
# coordA = (0.0, 0.0, 0.0)
# coordB = (0.0, 0.0, 1.0)
# # Create MaxG volumes.
# volA = genMaxGVol(MaxGDom(cellsA, scale, coordA))
# volB = genMaxGVol(MaxGDom(cellsB, scale, coordB))
# # Information for Green function construction. 
# # Complex frequency ratio. 
# freqPhase = 1.0 + im * 0.0
# # Gauss-Legendre approximation orders. 
# ordGLIntFar = 2
# ordGLIntMed = 4
# ordGLIntNear = 16
# # Cross over points for Gauss-Legendre approximation.
# crossMedFar = 16
# crossNearMed = 8
# assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
# 	ordGLIntNear, crossMedFar, crossNearMed)
# # Pre-allocate memory for circulant green function vector. 
# greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA[1], 2 * cellsA[2], 
# 	2 * cellsA[3])
# greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB[1] + cellsA[1], cellsB[2] +
# 	cellsA[2], cellsB[3] + cellsA[3])
# # CPU computation of Green function
# genGreenSlf!(greenCircAA, volA, assemblyInfo) # greenCircAA is modified here with the part of the Green function
# genGreenExt!(greenCircBA, volB, volA, assemblyInfo) # greenCircBA is modified here with the part of the Green function
# # You just have to compute the Green function once.
# # You can take the [*,*,:,:,:] to get a specific part of the total Green function.
# # By posing both * = 1, you get xx.
# print("genGreenExt[:,3,:,:,:] ", greenCircBA[:,3,:,:,:], "\n")
# # r=1 in this code is equivalent to r=2pi in G fancy function 
	

# I'm assuming we are going to do a linear solve for each direction in space? 
# s=floor(Int,sqrt(length(M)))
# chi_coeff = 3.0+2.0im
# chi = chi_coeff*Matrix(I, s, s)
# chi_invdag = inv(conj!(transpose(chi)))

# Get the g_{xz} using greenCircBA[1,3,:,:,:] and leave it as a 3x3 matrix. Calculate the inverse and the dag of G.
# Generate the chi matrix with 3+0.01j as a coefficient multiplied by the identity matrix. Calculate the inverse of this matrix. 
# Substract the results of step 1 and 2.
# Multiply the previous result by a complex P matrix. Let's say the complex identity matrix.
# Store the symmetric and asymmetric parts of the result of step 4.
# Add the lambda_1* the sym part and lambda_2* the asym part to get A.
# What would the initial v (variable you used in our discussion) be ? ji? What would ji be then?
# Once you have v and A, you can do all the steps (turning A into a vector using linear indexing, embedding, fft, element-wise multiplication of A with v and projection) to get your b vector. 
# Put the A matrix and b vector as inputs into the gmres algorithm.
# Redo steps 1-9 for the y-z and z-d directions. Then add all three results to get the result in the z-direction.
# Repeat steps 1-10 for the x- and y-directions.


# Code to test if the Green function works properly
cellsA_test = [1, 1, 1]
cellsB_test = [1, 1, 1]
# Size length of a cell relative to the wavelength. 
scale_test = 0.0001
# Center position of the volume. 
coordA_test = (0.0, 0.0, 0.0)
coordB_test = (4.3, 2.3, 1.1) # (0.0, 0.0, 1.0)
# Create MaxG volumes.
volA = genMaxGVol(MaxGDom(cellsA_test, scale_test, coordA_test))
volB = genMaxGVol(MaxGDom(cellsB_test, scale_test, coordB_test))
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
assemblyInfo = MaxGAssemblyOpts(freqPhase, ordGLIntFar, ordGLIntMed, 
	ordGLIntNear, crossMedFar, crossNearMed)
# Pre-allocate memory for circulant green function vector. 
greenCircAA = Array{ComplexF64}(undef, 3, 3, 2 * cellsA_test[1], 2 * cellsA_test[2], 
	2 * cellsA_test[3])
greenCircBA = Array{ComplexF64}(undef, 3, 3, cellsB_test[1] + cellsA_test[1], cellsB_test[2] +
	cellsA_test[2], cellsB_test[3] + cellsA_test[3])
# CPU computation of Green function
genGreenExt!(greenCircBA, volB, volA, assemblyInfo)
print("test ", greenCircBA[:,3,:,:,:],"\n")
# Incident field
# For the test we are only considering the z-direction, so we get a current density 
# vector that is a column vector of the form (0, 0, 1).
# Set an intial current density
ji_test = ones(ComplexF64, (cellsA_test[1])*(cellsA_test[2])*(cellsA_test[3]), 1) #double check if this is the right size of the initial ji
# Embed ji
ji_embedded_test = embedVec(ji_test, cellsA_test[1], cellsA_test[2], cellsA_test[3], cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3])
print("ji ", size(ji_embedded_test), "\n")

# z direction 
# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[1,3,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxz as a vector using linear indexing
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# print("g_test ", g_test, "\n")
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_xz = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is yz for this part of code
g_BA = greenCircBA[2,3,:,:,:] # this will get the entries for the z column of the Green's function
# Get gyz as a vector using linear indexing 
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# print("g_test ", g_test, "\n")
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_yz = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[3,3,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxx as a vector
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# Take the fft of ji_embedded (TBD what type of fft should be used)
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_zz = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

tot = projection_BA_xz + projection_BA_yz + projection_BA_zz
print("Tot in z " , tot, "\n")
print("projection_BA_xz " , projection_BA_xz, "\n")
print("projection_BA_yz " , projection_BA_yz, "\n")
print("projection_BA_zz " , projection_BA_zz, "\n")



# y direction 
# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[1,2,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxz as a vector using linear indexing
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_xy = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is yz for this part of code
g_BA = greenCircBA[2,2,:,:,:] # this will get the entries for the z column of the Green's function
# Get gyz as a vector using linear indexing 
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# print("g_test ", g_test, "\n")
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_yy = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[3,2,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxx as a vector
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# Take the fft of ji_embedded (TBD what type of fft should be used)
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_zy = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

tot = projection_BA_xy + projection_BA_yy + projection_BA_zy
print("Tot in x " , tot, "\n")
print("projection_BA_xy " , projection_BA_xy, "\n")
print("projection_BA_yy " , projection_BA_yy, "\n")
print("projection_BA_zy " , projection_BA_zy, "\n")


# x direction 
# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[1,1,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxz as a vector using linear indexing
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
print("g_test ", g_test, "\n")
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_xx = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is yz for this part of code
g_BA = greenCircBA[2,1,:,:,:] # this will get the entries for the z column of the Green's function
# Get gyz as a vector using linear indexing 
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# print("g_test ", g_test, "\n")
# Take the fft of ji_embedded 
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_yx = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

# Get the part of the total Green function we want, which is xz for this part of code
g_BA = greenCircBA[3,1,:,:,:] # this will get the entries for the z column of the Green's function
# Get gxx as a vector
g_test = zeros(ComplexF64, length(ji_embedded_test), 1)
index = 1
for element in g_BA
	if index < length(ji_embedded_test)
		g_test[index] = element
		global index += 1
	end
end
# Take the fft of ji_embedded (TBD what type of fft should be used)
x=rand(ComplexF64, length(ji_embedded_test), 1) 
p_BA = plan_fft(x) # ; flags=FFTW.ESTIMATE, timelimit=Inf
fft_ji_embedded_test = p_BA*ji_embedded_test
fft_g_test = p_BA*g_test
# Generate the circulent Green function and multiply it  
# by the fft of the embeddent source current 
mult_test = fft_g_test .* fft_ji_embedded_test
# Inverse the result of the previous multiplication
p_inv_test = plan_ifft(x; flags=FFTW.ESTIMATE, timelimit=Inf)
fft_inv_test = p_inv_test*mult_test
# Project the previous result 
projection_BA_zx = proj(fft_inv_test, cellsA_test[1]+cellsB_test[1], cellsA_test[2]+cellsB_test[2], cellsA_test[3]+cellsB_test[3], cellsB_test[1], cellsB_test[2], cellsB_test[3])

tot = projection_BA_xx + projection_BA_yx + projection_BA_zx
print("Tot in z " , tot, "\n")
print("projection_BA_xx " , projection_BA_xx, "\n")
print("projection_BA_yx " , projection_BA_yx, "\n")
print("projection_BA_zx " , projection_BA_zx, "\n")


