import numpy as np

# First constraint
def C1(T,i,ei, chi_invdag, Gdag,Pv):
    # Left term
    P=Pv[i]
    # print("P shape",P.shape,"\n")
    # print("T shape",T,"\n")
    # PT=np.matmul(P,T)
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT = np.vdot(ei, P.dot(T))
    # EPT = np.dot(ei_tr,PT)
    I_EPT = np.imag(EPT) # (1/2)*
    # Right term
    chiGP = (chi_invdag - Gdag)@P
    # np.matmul(P,chi_invdag - Gdag)
    chiG_A = np.imag(chiGP)
    # (chiGP-np.matrix.conjugate(np.transpose(chiGP)))/(2j)
    # M^A = (M+M^dagg)/2 -> j is i in python
    # chiGA_T = np.matmul(chiG_A,T)
    # T_chiGA_T = np.matmul(np.matrix.conjugate(np.transpose(T)),chiGA_T)
    T_chiGA_T = np.real(np.vdot(T, chiG_A.dot(T)))
    # T_chiGA_T = np.real(np.vdot(T, gradZTT[i].dot(T)))
    
    # print("I_EPT A", I_EPT, "\n")
    # print("T_chiGA_T", T_chiGA_T, "\n")
    return I_EPT - T_chiGA_T
# Second constraint
def C2(T,i,ei, chi_invdag, Gdag,Pv):
    # Left term
    P=Pv[i]
    # PT=np.matmul(P,T)
    # E_tc = np.matrix.conjugate(np.transpose(ei))
    EPT = np.vdot(ei, P.dot(T))
    # EPT = np.dot(ei_tr,PT)
    I_EPT = np.real(EPT) # (1/2)*
    # Right term
    chiGP = (chi_invdag - Gdag)@P
    # np.matmul(P,chi_invdag - Gdag)
    # M^S = (M+M^dagg)/2 
    chiG_S = np.real(chiGP)
    #(chiGP+np.matrix.conjugate(np.transpose(chiGP)))/2
    # chiGS_T = np.matmul(chiG_S,T)
    T_chiGS_T = np.real(np.vdot(T, chiG_S.dot(T)))
    # T_chiGS_T = np.matmul(np.matrix.conjugate(np.transpose(T)),chiGS_T)

    # print("I_EPT S", I_EPT, "\n")
    # print("T_chiGS_T", T_chiGS_T, "\n")
    return I_EPT - T_chiGS_T