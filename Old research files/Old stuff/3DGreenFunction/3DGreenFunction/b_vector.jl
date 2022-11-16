module b_vector
export bv
# creation of b 
function bv(ei, l, P) 
    return -ei/(2im) + (l[2]/(2im))*P*ei + (l[2]/2)*P*ei
    # just like for A, the first lambda is associated 
    # with the asym part and the second lambda is 
    # associated with the sym part 
end
end