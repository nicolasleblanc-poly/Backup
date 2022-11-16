module vector
export vect
# turn the Green functions into vectors using linear indexing
i=1
function vect(g)
    # i = 1
    len_g = length(g)
    g_vector = zeros(ComplexF64, len_g, 1)
    while i < len_g
        g_vector[i] = g[i]
        global i += 1
    end

    # for element in g
    #     if index < len_g
    #         g_vector[index] = element
    #         global index += 1
    #     end
    # end
    return g_vector
end
end