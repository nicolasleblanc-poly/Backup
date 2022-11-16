using Distributed
@everywhere using Printf, Base.Threads, MaxGParallelUtilities, MaxGStructs, MaxGBasisIntegrals, MaxGCirc, LinearAlgebra, Embedding, Projection, FFTW, vector, G_v_product, b_vector, Gdag_v_product, A_lin_op, bfgs
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
ei = []
for cellZ = 1:cellsB[3]
    for cellY = 1:cellsB[2]
        for cellX = 1:cellsB[1]
            global s = 2 * pi * sqrt((cellX-seperation)^2 + cellY^2 + cellZ^2)
            global sh = (2 * pi) .* ((cellX-seperation), cellX, cellZ) ./ s 
            # operators
            # let's consider a dipole oriented in the z-direction
            id = [0.0 0.0 1.0]
            sHs = [(sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]
            # id = [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 1.0]
            # sHs = [(sh[1] * sh[1]) (sh[1] * sh[2]) (sh[1] * sh[3]); (sh[2] * sh[1]) (sh[2] * sh[2]) (sh[2] * sh[3]); (sh[3] * sh[1]) (sh[3] * sh[2]) (sh[3] * sh[3])]
            # analytic green function

            gAna = (exp(im * s) / (4 * pi * s)) .* (((1 + (im * s - 1) / s^2) .* id) .- ((1 + 3 * (im * s - 1) / s^2) .* sHs))
            append(ei,gAna)
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
P = I
# Pdag
# Pdag = conj.(transpose(P)) 
# we could do the code above for a P other than the identity matrix
Pdag = P
# let's get the initial b vector (aka using the initial Lagrange multipliers). Done for test purposes
b = bv(ei, l,P)
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
# let's get the output (Green function?) in the x, in the y and in the z by calculating G|v>, where |v> is b 
# xx ouput 
o_xx = G_v(g_xx, b, cellsA)
# xy ouput
o_xy = G_v(g_xy, b, cellsA)
# xz ouput
o_xz = G_v(g_xz, b, cellsA)
# yx ouput
o_yx = G_v(g_yx, b, cellsA)
# yy ouput
o_yy = G_v(g_yy, b, cellsA)
# yz ouput
o_yz = G_v(g_yz, b, cellsA)
# zx ouput
o_zx = G_v(g_zx, b, cellsA)
# zy ouput
o_zy = G_v(g_zy, b, cellsA)
# zz ouput
o_zz = G_v(g_zz, b, cellsA)
# let's get the output in each direction of space (x, y and z)
o_x = o_xx + o_xy + o_xz 
o_y = o_yx + o_yy + o_yz 
o_z = o_zx + o_zy + o_zz 

# let's do the samething but now to get the dag outputs
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
# let's get the output (Green function?) in the x, in the y and in the z by calculating G|v>, where |v> is b 
# xx ouput 
o_xx_dag = G_v(g_xx_dag, b, cellsA)
# xy ouput
o_xy_dag = G_v(g_xy_dag, b, cellsA)
# xz ouput
o_xz_dag = G_v(g_xz_dag, b, cellsA)
# yx ouput
o_yx_dag = G_v(g_yx_dag, b, cellsA)
# yy ouput
o_yy_dag = G_v(g_yy_dag, b, cellsA)
# yz ouput
o_yz_dag = G_v(g_yz_dag, b, cellsA)
# zx ouput
o_zx_dag = G_v(g_zx_dag, b, cellsA)
# zy ouput
o_zy_dag = G_v(g_zy_dag, b, cellsA)
# zz ouput
o_zz_dag = G_v(g_zz_dag, b, cellsA)
# let's get the output in each direction of space (x, y and z)
o_x_dag = o_xx_dag + o_xy_dag + o_xz_dag 
o_y_dag  = o_yx_dag + o_yy_dag + o_yz_dag 
o_z_dag  = o_zx_dag + o_zy_dag + o_zz_dag 
# we use these outputs as the G in objective and constraints of Sean's articles.

# let's work firstly in the x-direction
# chi^{-1}-G 
chi_g_x = chi_inv_coeff .- o_x
# chi^{-1 dag}-G^{dag} 
chi_g_dag_x = chi_inv_coeff_dag .- o_x_dag
# calculation of the asymmetric and symmetric parts 
# of the constraints
# asymmetric part 
asym_x = (1\(2im))*(P*chi_g_dag_x-chi_g_dag_x*Pdag)
# symmetric part
sym_x = (1\2)*(P*chi_g_dag_x+chi_g_dag_x*Pdag)
# calculate the vacuum contribution
e_vac = 0 #check it's value with Sean

l_x = [1;0.1] # initial Lagrange multipliers
bfgs_x = BFGS_fakeS_with_restart(l_x,dual,ei,e_vac,asym_x,sym_x,cellsA,validityfunc, mineigfunc)

dof = bfgs_x[1]
grad = bfgs_x[2]
dualval = bfgs_x[3]
objval = bfgs_x[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")


# now the y-direction
# let's work firstly in the x-direction
# chi^{-1}-G 
chi_g_y = chi_inv_coeff .- o_y
# chi^{-1 dag}-G^{dag} 
chi_g_dag_y = chi_inv_coeff_dag .- o_y_dag
# calculation of the asymmetric and symmetric parts 
# of the constraints
# asymmetric part 
asym_y = (1\(2im))*(P*chi_g_dag_y-chi_g_dag_y*Pdag)
# symmetric part
sym_y = (1\2)*(P*chi_g_dag_y+chi_g_dag_y*Pdag)
# calculate the vacuum contribution
e_vac = 0 #check it's value with Sean

l_y = [1;0.1] # initial Lagrange multipliers
bfgs_y = BFGS_fakeS_with_restart(l_y,dual,ei,e_vac,asym_y,sym_y,cellsA,validityfunc, mineigfunc)

dof = bfgs_y[1]
grad = bfgs_y[2]
dualval = bfgs_y[3]
objval = bfgs_y[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")



# and finally the z-direction
# let's work firstly in the x-direction
# chi^{-1}-G 
chi_g_z = chi_inv_coeff .- o_z
# chi^{-1 dag}-G^{dag} 
chi_g_dag_z = chi_inv_coeff_dag .- o_z_dag
# calculation of the asymmetric and symmetric parts 
# of the constraints
# asymmetric part 
asym_z = (1\(2im))*(P*chi_g_dag_z-chi_g_dag_z*Pdag)
# symmetric part
sym_z = (1\2)*(P*chi_g_dag_z+chi_g_dag_z*Pdag)
# calculate the vacuum contribution
e_vac = 0 #check it's value with Sean

l_z = [1;0.1] # initial Lagrange multipliers
bfgs_z = BFGS_fakeS_with_restart(l_z,dual,ei,e_vac,asym_z,sym_z,cellsA,validityfunc, mineigfunc)

dof = bfgs_z[1]
grad = bfgs_z[2]
dualval = bfgs_z[3]
objval = bfgs_z[4]
print("dof ", dof, "\n")
print("grad ", grad, "\n")
print("dualval ", dualval, "\n")
print("objval ", objval, "\n")