# Coordinates of B
x = 0
y = 1
z = 1

# Coefficient
r = 2*pi*sqrt(x^2+y^2+z^2) # since there is a factor of 2pi between the 3D Green function code and this function
coeff = exp(-r*1im)/(4*pi*r)

# First term calculation
id = zeros(Float64, 3, 1)
id[3] = 1
t1 = (1 + (r*1im-1)/r^2)*id

# Second term calculation
coeff_2 = 1 + (3(r*1im-1))/r^2

# Calculate the dyadic product
# A is centered at (0,0,0) and B is centered at (0,0,0)
theta = 0.0916308344441178
phi = pi/4
# 0.0916308344441178
# theta = atan(z/r)
# phi = atan(y/x)
# print("theta: ", theta, "\n")
# print("phi: ", phi, "\n")

rx = (sin(theta)*cos(phi))
ry = (sin(theta)*sin(phi))
rz = cos(theta)

# Let's store the z-column of the dyadic product matrix 
# in a column vector 
vec = zeros(Float64, 3, 1)
vec[1] = rx*rz
vec[2] = ry*rz
vec[3] = rz*rz

t2 = coeff_2*vec 

# Put everything together
res = coeff*(t1-t2)
print("res", res, "\n")

