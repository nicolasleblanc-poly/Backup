# import Pkg
# Pkg.add("MPI")
# Pkg.add("PencilFFTs")

using MPI
using PencilFFTs

MPI.Init()
comm = MPI.COMM_WORLD

# Input data dimensions (Nx × Ny × Nz)
dims = (2,2,2)
pen = Pencil(dims, comm)

# Apply a 3D real-to-complex (r2c) FFT.
transform = Transforms.FFT()

# Note that, for more control, one can instead separately specify the transforms along each dimension:
# transform = (Transforms.RFFT(), Transforms.FFT(), Transforms.FFT())

# Create plan
plan = PencilFFTPlan(pen, transform)

# In our example, this returns a 3D PencilArray of real data (Float64).
u = allocate_input(plan)

# Fill the array with some (random) data
using Random
randn!(u)

# In our example, this returns a 3D PencilArray of complex data (Complex{Float64}).
v = allocate_output(plan)

using LinearAlgebra  # for mul!, ldiv!

# Apply plan on `u` with `v` as an output
mul!(v, plan, u)

# Apply backward plan on `v` with `w` as an output
w = similar(u)
ldiv!(w, plan, v)  # now w ≈ u
print("input ", u, "\n")
print("output ", w, "\n")
