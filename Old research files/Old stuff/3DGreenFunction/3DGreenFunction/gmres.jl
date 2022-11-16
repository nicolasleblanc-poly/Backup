module gmres
export GMRES_with_restart_x
using G_v_product, LinearAlgebra, vector
# based on the example code from https://tobydriscoll.net/fnc-julia/krylov/gmres.html
# code for the AA case 
i = 1
# m is the maximum number of iterations
function GMRES_with_restart_x(l,g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag, b, cellsA, m=20) # add cellsB for the BA case 
    n = length(b)
    # print("size(b) ", size(b), "\n")
    Q = zeros(ComplexF64,n,m+1)
    Q[:,1] = b/norm(b)
    H = zeros(ComplexF64,m+1,m)
    # Initial solution is zero.
    x = 0
    residual = [norm(b);zeros(m)]
    for j in 1:m
        # first G|v> type calculation
        g_output = output(g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,Q[:,j],cellsA)
        o_x = g_output[1] # total x output 
        o_x_dag = g_output[2] # total x dag output  
        # let's define some values
        # chi coefficient
        chi_coeff = 3.0 + 0.01im
        # inverse chi coefficient
        chi_inv_coeff = 1/chi_coeff 
        chi_inv_dag_coeff = conj(chi_inv_coeff)
        # define the projection operators
        # I didn't include P in the calculation since P = I for this case
        term1 = chi_inv_dag_coeff*Q[:,j]
        term2 = o_x_dag
        term3 = chi_inv_coeff*Q[:,j]
        term4 = o_x
        v = l[1]*(1\(2im))*(term1-term2-term3+term4)
        # v = G_v(A,Q_temp,cellsA)
        
        for i in 1:j
            H[i,j] = dot(Q[:,i],v)
            v -= H[i,j]*Q[:,i]
        end
        # print("Here \n")
        H[j+1,j] = norm(v)
        Q[:,j+1] = v/H[j+1,j]
        # Solve the minimum residual problem.
        r = [norm(b); zeros(ComplexF64,j)]
        z = H[1:j+1,1:j] \ r
        x = Q[:,1:j]*z

        # second G|v> type calculation
        g_output = output(g_xx, g_xy, g_xz, g_xx_dag, g_xy_dag, g_xz_dag,x,cellsA)
        o_x = g_output[1] # total x output 
        o_x_dag = g_output[2] # total x dag output   
        # we'll use the values defined above
        term1 = chi_inv_dag_coeff*x
        term2 = o_x_dag
        term3 = chi_inv_coeff*x
        term4 = o_x
        value = l[1]*(1\(2im))*(term1-term2-term3+term4)
        residual[j+1] = norm(value - b )
        # residual[j+1] = norm(G_v(A,x,cellsA) - b )
    end
    return x
end
end