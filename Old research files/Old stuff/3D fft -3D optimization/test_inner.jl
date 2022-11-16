using FFTW
# 2x2x2 -> 4x4x4
a = ones(ComplexF64, 4, 1)
# test using dims
x1=rand(ComplexF64, 4, 1) 
x2=rand(ComplexF64, 2, 1) 
# test not using dims and doing it by hand

"""
for a nxnxn domain, you'll be taking the fft's of 2nx1 vectors, right?
x: fft of an 4x1 vector that is made of the (1,2,3,4) elements of a and the repeat for a vector made of
the next 4 elements of the vector a -> steps of 1, so we get 16 fft's of 4x1 vectors
y: fft of an 4x1 vector that is made of the (1,5,9,13) or (1,5,9,13,17,21,25,29,33,37,...) and 
repeat with the other elements of the vector made up of the results of the fft's in the x 
-> steps of 4, so we get 16 fft's of 4x1 vectors
z: fft of an 4x1 vector that is made of the (1,9,17,25) or (1,9,17,25,...) and repeat 
with the other elements of the vector made up of the results of the fft's in the y
-> steps of 8, so we get 16 fft's of 4x1 vectors 

do we just need one plan then or 3 plans because it seems that we only need one if 
we are always considering that we are taking the fft of 4x1 vectors but you said 
that we would need 3 fft plans, so I am confused. 
The dims parameter of the fft_plan function is kinda confusing because 
if we want to take the fft of the rows of the vectors we use dims=2, so 
what would we use if we wanted to take the fft with a step? Would 
we have dims = 4 for a step = 4 and dims = 16 for a step of 16?
"""

# calculation of the direct fft plan 

# dims fft calculation
fft_plan_2 = plan_fft(x1,2, flags=FFTW.ESTIMATE, timelimit=Inf) 
# fft_plan_3 = plan_fft(x,3, flags=FFTW.ESTIMATE, timelimit=Inf) 

print("dims fft ", fft_plan_2*a, "\n")

# no dims fft calculation
fft_plan = plan_fft(x2; flags=FFTW.ESTIMATE, timelimit=Inf) 

v1 = zeros(ComplexF64, 2, 1)
v1[1] = a[1]
v1[2] = a[2]
v2 = zeros(ComplexF64, 2, 1)
v2[1] = a[3]
v2[2] = a[4]

res_1 = fft_plan*v1
res_2 = fft_plan*v2
a_2 = zeros(ComplexF64, 4, 1)
a_2[1] = res_1[1]
a_2[2] = res_1[2]
a_2[3] = res_2[1]
a_2[4] = res_2[2]

u1 = zeros(ComplexF64, 2, 1)
u1[1] = a_2[1]
u1[2] = a_2[3]
u2 = zeros(ComplexF64, 2, 1)
u2[1] = a_2[2]
u2[2] = a_2[4]

calc_1 = fft_plan*u1
calc_2 = fft_plan*u2
res = zeros(ComplexF64, 4, 1)
res[1] = res_1[1]
res[2] = res_1[2]
res[3] = res_2[1]
res[4] = res_2[2]

print("no dims fft ", res, "\n")

# calculation of the inverse fft plan
inv_fft_plan = plan_ifft(x2; flags=FFTW.ESTIMATE, timelimit=Inf)

# calculation of the direct fft plan 
# fft_plan_dims_x = plan_fft(x, 8, flags=FFTW.ESTIMATE, timelimit=Inf) 
# fft_plan_dims_y = plan_fft(x, 2, flags=FFTW.ESTIMATE, timelimit=Inf) 
# fft_plan_dims_z = plan_fft(x, 2, flags=FFTW.ESTIMATE, timelimit=Inf) 
# calculation of the inverse fft plan
# inv_fft_plan_dims = plan_ifft(x, 2, flags=FFTW.ESTIMATE, timelimit=Inf)

# fft_dims = fft_plan_dims*a
# print("fft_dims ", fft_dims, "\n")