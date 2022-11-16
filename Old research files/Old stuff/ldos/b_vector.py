import math
import numpy as np
# Code for the creation of the b vector used to solve for T in the Ax=b system.
def bv(x,ei,Pv, e_vac):
        term1 = -1/(2j) # k0*1.0j/(4*Z)
        b = term1*ei 
        # print("b1", b, "\n")
        for i in range(len(x)):
            P=Pv[math.floor(i/2)]
            #term = (1/2)*x[i]*P
            #b += np.matmul(term,ei)
            if i % 2 ==0: # Anti-sym
                term = (1/2j)*x[i]*P
                b += np.matmul(term,ei)
            else: # Sym
                term = (1/2)*x[i]*P
                b += np.matmul(term,ei)
            # print("b2", b, "\n")
        return b