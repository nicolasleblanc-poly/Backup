module b_asym_only
export bv
# creation of b 
function bv(ei, l, P) 
    # print("-ei/(2im) ", -ei/(2im), "\n")
    # print("(l[1]/(2im))*P*ei  ", (l[1]/(2im))*P*ei , "\n")
    return -ei/(2im) + (l[1]/(2im))*P*ei 
    # just like for A, the first lambda is associated 
    # with the asym part and the second lambda is 
    # associated with the sym part 
end
end