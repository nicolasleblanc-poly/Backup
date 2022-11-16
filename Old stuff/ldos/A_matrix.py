import numpy as np
import math
# Code for the creation of the A matrix
def Am(x, chi_invdag, Gdag,Pv): # the A matrix
    A=0.0+0.0j
    for i in range(len(x)):
        P = Pv[math.floor(i/2)]
        chiGP = (chi_invdag - Gdag)@P # np.matmul(P,chi_invdag - Gdag)
        # print("chiGP", chiGP, "\n")
        if i % 2 == 0: # Anti-sym
            chiG_A = np.imag(chiGP)
            #(chiGP-np.matrix.conjugate(np.transpose(chiGP)))/(2j)
            A += x[i]*chiG_A
        else: # Sym
            chiG_S = np.real(chiGP)
            # (chiGP+np.matrix.conjugate(np.transpose(chiGP)))/(2)
            A += x[i]*chiG_S
        # print("A", A, "\n")
    return A