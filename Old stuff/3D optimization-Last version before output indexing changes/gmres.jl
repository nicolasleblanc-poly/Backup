module gmres
export GMRES_with_restart
using G_v_product, LinearAlgebra, vector
# based on the example code from https://tobydriscoll.net/fnc-julia/krylov/gmres.html
# code for the AA case 
i = 1
# m is the maximum number of iterations
function GMRES_with_restart(l, b, fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA, m=20) # add cellsB for the BA case 
    n = length(b)
    # print("size(b) ", size(b), "\n")
    Q = zeros(ComplexF64,n,m+1)
    # print("b ", b, "\n")
    Q[:,1] = b/norm(b)
    H = zeros(ComplexF64,m+1,m)
    # Initial solution is zero.
    x = 0
    residual = [norm(b);zeros(m)]
    for j in 1:m
        # first G|v> type calculation
        v = output(l,Q[:,j],fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
        # print("Q[:,j] ", Q[:,j], "\n")
        
        # print("v ", v, "\n")
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            v -= H[i,j]*Q[:,i]
        end
        

        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]
        # Solve the minimum residual problem.
        r = [norm(b); zeros(ComplexF64,j)]
        z = H[1:j+1,1:j] \ r
        x = Q[:,1:j]*z
        # second G|v> type calculation
        value = output(l,x,fft_plan,inv_fft_plan,g_xx,g_xy,g_xz,g_yx,g_yy,g_yz,g_zx,g_zy,g_zz,cellsA)
        residual[j+1] = norm(value - b )
    end
    return x
end
end