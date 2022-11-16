using Distributed
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, Embedding, Projection, FFTW, vector, G_v_product, b_asym_only, Gdag_v_product, A_asym_only, bfgs_asym_only, gmres, dual_asym_only
# Set number of BLAS threads. The number of Julia threads is set as an 
# environment variable. The total number of threads is Julia threads + BLAS 
# threads. VICu is does not call BLAS libraries during threaded operations, 
# so both thread counts can be set near the available number of cores. 
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
ei_x = []
ei_y = []
ei_z = []
for cellZ = 1:cellsA[3]
    for cellY = 1:cellsA[2]
        for cellX = 1:cellsA[1]
            global s = 2 * pi * sqrt((cellX-seperation)^2 + cellY^2 + cellZ^2)
            global sh = (2 * pi) .* ((cellX-seperation), cellX, cellZ) ./ s 
            # operators
            # let's consider a dipole oriented in the z-direction
            id = [0.0 0.0 1.0]
            sHs = [(sh[1] * sh[3]);  (sh[2] * sh[3]); (sh[3] * sh[3])] #xz, yz and zz
            # analytic green function
            gAna = (exp(im * s) / (4 * pi * s)) .* (((1 + (im * s - 1) / s^2) .* id) .- ((1 + 3 * (im * s - 1) / s^2) .* sHs))
            append!(ei_x,gAna[1])
            append!(ei_y,gAna[2])
            append!(ei_z,gAna[3])
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
b_x = bv(ei_x, l,P)
b_y = bv(ei_y, l,P)
b_z = bv(ei_z, l,P)

# print("size(b) ", size(b), "\n")

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
# xx 
g_xx = vect(gf_xx)
# xy
g_xy = vect(gf_xy)
# xz 
g_xz = vect(gf_xz)
# yx
g_yx = vect(gf_yx)
# yy 
g_yy = vect(gf_yy)
# yz 
g_yz = vect(gf_yz)
# zx 
g_zx = vect(gf_zx)
# zy
g_zy = vect(gf_zy)
# zz 
g_zz = vect(gf_zz)
# let's do the samething but now to get the gdag outputs
# dag Green functions
# xx 
g_xx_dag = conj.(g_xx)
# xy
g_xy_dag = conj.(g_xy)
# xz 
g_xz_dag = conj.(g_xz)
# yx
g_yx_dag = conj.(g_yx)
# yy 
g_yy_dag = conj.(g_yy)
# yz 
g_yz_dag = conj.(g_yz)
# zx 
g_zx_dag = conj.(g_zx)
# zy
g_zy_dag = conj.(g_zy)
# zz 
g_zz_dag = conj.(g_zz)

e_vac = 0 #check it's value with Sean
# x-direction
l_x = [2] # initial Lagrange multipliers
bfgs_x = BFGS_fakeS_with_restart_x(l_x,dual,P,ei_x,e_vac,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA)
dof = bfgs_x[1]
grad = bfgs_x[2]
dualval = bfgs_x[3]
objval = bfgs_x[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")
# now the y-direction
l_y = [2] # initial Lagrange multipliers
bfgs_y = BFGS_fakeS_with_restart_x(l_y,dual,P,ei_y,e_vac,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA)
dof = bfgs_y[1]
grad = bfgs_y[2]
dualval = bfgs_y[3]
objval = bfgs_y[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")
# and finally the z-direction
# x-direction
l_z = [2] # initial Lagrange multipliers
bfgs_z = BFGS_fakeS_with_restart_x(l_z,dual,P,ei_z,e_vac,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,cellsA)
dof = bfgs_z[1]
grad = bfgs_z[2]
dualval = bfgs_z[3]
objval = bfgs_z[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")