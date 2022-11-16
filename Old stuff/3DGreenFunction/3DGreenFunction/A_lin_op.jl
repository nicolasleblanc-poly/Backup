module A_lin_op
export A_op
# creation of A linear operation
function A_op(l,asym,sym) 
    # calculate the symmetric and asymmetric parts of A 
    A = l[1]*asym + l[2]*sym 
    # the first lambda is associated with the asym part
    # the second lambda is assiciated with the sym part 
    return A
end
end