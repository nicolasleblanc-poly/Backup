import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la


def get_etalist(G, Td, O, Plist):
    """
    get the factors of eta that multiply the global convex constraint to ensure semidefiniteness of the objective and other constraints
    """

    GTd = G @ Td
    AsymTd = (Td-Td.conj().T)/2j
    L = la.cholesky(AsymTd)
    Linv = la.solve_triangular(L, np.eye(L.shape[0]))

    eta_O = max(0, -la.eigvalsh(Linv.conj().T @ O @ Linv)[0])
    
    etalist = [] #in the order of eta_s+, eta_s-, eta_h+ for each P
    for i in range(len(Plist)):
        P = Plist[i]
        PGTd = P @ GTd
        SymPGTd = (PGTd + PGTd.conj().T)/2
        AsymPGTd = (PGTd - PGTd.conj().T)/2j

        eigsL_AsymPGTd = la.eigvalsh(Linv.conj().T @ AsymPGTd @ Linv)
        eta_sp = max(0, -eigsL_AsymPGTd[0])
        eta_sm = max(0, eigsL_AsymPGTd[-1])

        eta_hp = max(0, -la.eigvalsh(Linv.conj().T @ (P+SymPGTd) @ Linv)[0])

        etalist.extend([eta_sp, eta_sm, eta_hp])

    return eta_O, etalist


def get_eta_O_gradLops(chi, y, G, O_lin, O_quad, Plist, Td=None):
    
    if Td is None:
        Td = la.inv((1.0/chi)*np.eye(G.shape[0]) - G) #see what we can do in future about this explicit matrix inverse
    GTd = G @ Td
    Td_y = Td @ y
    AsymTd = (Td-Td.conj().T)/2j
    
    L = la.cholesky(AsymTd)
    Linv = la.solve_triangular(L, np.eye(L.shape[0]))

    eta_O = max(0, -la.eigvalsh(Linv.conj().T @ O_quad @ Linv)[0])
    O_quad_eta = O_quad + eta_O*AsymTd
    O_lin_eta = O_lin + eta_O*(-0.5j)*Td_y
    
    etalist = [] #in the order of eta_s+, eta_s-, eta_h+, eta_h- for each P
    gradLzz = []
    gradLzy_y = []
    for i in range(len(Plist)):
        P = Plist[i]
        PpPGTd = P + P@GTd
        SymPpPGTd = (PpPGTd + PpPGTd.conj().T)/2
        AsymPpPGTd = (PpPGTd - PpPGTd.conj().T)/2j

        #get the eta factors
        eigsL_AsymPpPGTd = la.eigvalsh(Linv.conj().T @ AsymPpPGTd @ Linv)
        eta_sp = max(0, -eigsL_AsymPpPGTd[0])
        eta_sm = max(0, eigsL_AsymPpPGTd[-1])

        eigsL_SymPpPGTd = la.eigvalsh(Linv.conj().T @ SymPpPGTd @ Linv)
        eta_hp = max(0, -eigsL_SymPpPGTd[0])
        eta_hm = max(0, eigsL_SymPpPGTd[-1])
        

        etalist.extend([eta_sp, eta_sm, eta_hp, eta_hm])
        
        #get the linear and bilinear contributions
        PpPGTd_y = PpPGTd @ y
        gradLzy_y.append(-0.5j*(PpPGTd_y + eta_sp*Td_y))
        gradLzy_y.append(-0.5j*(-PpPGTd_y + eta_sm*Td_y))
        gradLzy_y.append(0.5*PpPGTd_y - 0.5j*eta_hp*Td_y)
        gradLzy_y.append(-0.5*PpPGTd_y - 0.5j*eta_hm*Td_y)
        
        
        gradLzz.append(AsymPpPGTd + eta_sp*AsymTd)
        gradLzz.append(-AsymPpPGTd + eta_sm*AsymTd)
        gradLzz.append(SymPpPGTd + eta_hp*AsymTd)
        gradLzz.append(-SymPpPGTd + eta_hm*AsymTd)
        
    #the global convex constraint
    gradLzy_y.append(-0.5j*Td_y)
    gradLzz.append(AsymTd)

    return eta_O, etalist, O_lin_eta, O_quad_eta, gradLzy_y, gradLzz
