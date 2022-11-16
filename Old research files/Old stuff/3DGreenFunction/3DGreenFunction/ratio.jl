# #101 case
# gval = 2.0835188596250243e-10 + 4.5798920612314135e-10im
# tval = -0.00415025834604437 - 0.0018950344398860977im
# print("101 ", gval/tval, "\n")

# #011 case
# gval = 2.0835188597605496e-10 + 4.579892061197532e-10im
# tval = -0.00415025834604437 - 0.0018950344398860977im
# print("011 ", gval/tval, "\n")

# #111 case
# gval = -3.3466166723164606e-10 + 7.078235038566506e-12im
# tval = -0.001278243349787237 + 0.0015547338674045679im
# print("111 ", gval/tval, "\n")

# # 001 case
# gval = 1.5915441946116882e-10 - 9.999967101747211e-10im
# tval = -0.0070341916800216 - 0.00535911865562291im
# print("001 ",gval/tval, "\n")

# using LinearAlgebra
# print([2 2] *I)

# i = 1
# g = ones(ComplexF64, 2, 2)
# len_g = length(g)
# g_vector = zeros(ComplexF64, 4, 1)
# while i < len_g
#     g_vector[i] = g[i]
#     global i += 1
# end
# print(g_vector)
using LinearAlgebra
# print(2*I-[2 2; 2 2])

A = zeros(ComplexF64, 4, 4)
A[1] = 1
A[6] = 1
A[11] = 1
A[16] = 1
A_new = 2*A
A_vec = ones(ComplexF64, 4, 1)
i = 1 
while i < len_g 
    A_vec[i] = A[i]
    global i += 1
end
g = ones(ComplexF64, 4, 1)
g[1] = 2
g[2] = 3
g[3] = 4
g[4] = 5
mult = A_new*g
print(mult)