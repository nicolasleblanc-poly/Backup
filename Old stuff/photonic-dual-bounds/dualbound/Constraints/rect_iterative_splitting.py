import numpy as np
import random
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .rect_domains import get_rect_proj_in_rect_region


def split_generalP_rect_domains(Pdof, ulco_list, brco_list, n_S=1):
    split_ulco_list=[]
    split_brco_list=[]
    split_Pdof=[]
    
    flag=False #indicates whether any region has been split or not
    for i in range(len(ulco_list)):
        ulco = ulco_list[i]
        brco = brco_list[i]
        lx = brco[0] - ulco[0]
        ly = brco[1] - ulco[1]

        projLagnum = 2*n_S**2
        if lx>=ly and lx>1:
            split_ulco_list.append(ulco)
            split_brco_list.append((ulco[0]+lx//2,brco[1]))
            
            split_ulco_list.append((ulco[0]+lx//2,ulco[1]))
            split_brco_list.append(brco)

            split_Pdof.extend(Pdof[i*projLagnum:(i+1)*projLagnum].tolist() * 2)
            flag=True
        elif ly>=lx and ly>1:
            split_ulco_list.append(ulco)
            split_brco_list.append((brco[0],ulco[1]+ly//2))

            split_ulco_list.append((ulco[0],ulco[1]+ly//2))
            split_brco_list.append(brco)

            split_Pdof.extend(Pdof[i*projLagnum:(i+1)*projLagnum].tolist() * 2)
            flag=True
        else: #no split was possible
            split_ulco_list.append(ulco)
            split_brco_list.append(brco)
            split_Pdof.extend(Pdof[i*projLagnum:(i+1)*projLagnum].tolist())

    if not flag:
        raise ValueError('no more subdivisions possible')

    return np.array(split_Pdof), split_ulco_list, split_brco_list


def get_rect_Ideslist(design_ulco, design_brco, proj_ulco_list, proj_brco_list, sparse=False, pol='TM'):
    Ideslist = []
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region(design_ulco, design_brco, proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        if sparse:
            Proj_mat = sp.diags(Proj_mat, format="csc")
            if pol=='TE':
                Proj_mat = sp.kron(sp.eye(2), Proj_mat, format='csc')
        else:
            Proj_mat = np.diag(Proj_mat)
            if pol=='TE':
                Proj_mat = np.kron(np.eye(2), Proj_mat)

        Ideslist.append(Proj_mat)
        
    return Ideslist

def get_rect_Ideslist_vacuum(design_mask, design_ulco, design_brco, proj_ulco_list, proj_brco_list, sparse=False, pol='TM'):
    Ideslist = []
    design_mask = design_mask.flatten()
    for i in range(len(proj_brco_list)):
        proj_ulco = proj_ulco_list[i]
        proj_brco = proj_brco_list[i]
        Proj_bool = get_rect_proj_in_rect_region(design_ulco, design_brco, proj_ulco, proj_brco)
        Proj_mat = np.zeros_like(Proj_bool, dtype=int)
        Proj_mat[Proj_bool] = 1
        Proj_mat = Proj_mat[design_mask]
        if sparse:
            Proj_mat = sp.diags(Proj_mat, format="csc")
            if pol=='TE':
                Proj_mat = sp.kron(sp.eye(2), Proj_mat, format='csc')
        else:
            Proj_mat = np.diag(Proj_mat)
            if pol=='TE':
                Proj_mat = np.kron(np.eye(2), Proj_mat)

        Ideslist.append(Proj_mat)
        
    return Ideslist


def dualopt_iterative_splitting(n_S, Mx, My, U, dualoptFunc, outputFunc=None, pol='TM', reducedBasis=None):
    """
    runs dual optimization while iteratively increasing the number of projection constraints, seeding later optimizations with solution of prior optimizations
    dualoptfunc(n_S, initLags, Plist, UPlist, include) returns dual optimum, optimizing from initLags
    """

    design_ulco = (0,0)
    design_brco = (Mx,My)
    subproj_ulco_list = [(0,0)]
    subproj_brco_list = [(Mx,My)] #start with just the global constraints
    num_iters = 0
    while True:
        num_iters += 1
        proj_ulco_list = [design_ulco] + subproj_ulco_list
        proj_brco_list = [design_brco] + subproj_brco_list

        Plist = get_rect_Ideslist(design_ulco, design_brco, proj_ulco_list, proj_brco_list, pol=pol)
        num_region = len(Plist)
        print('Total # of subregions', num_region-1, flush=True)
        UPlist = []
        for i in range(num_region):
            UPlist.append(U @ Plist[i])

        if not (reducedBasis is None): #dimension reduction for affine constraints
            for i in range(num_region):
                UPlist[i] = reducedBasis.conj().T @ (UPlist[i] @ reducedBasis)
                Plist[i] = Plist[i] @ reducedBasis
            
        #####SET THE INCLUDE LIST#####
        include = np.array([True] * (num_region * 2 * n_S**2))
        include[2*n_S**2:4*n_S**2] = False #since we include global constraints, leave out first subregion to prevent linear dependence

        if num_iters==1: #start of iteration, initialize Lags so only global Asym constraints in each source have non-zero multipliers
            Lags = np.zeros(len(include))
            for i in range(n_S):
                Lags[n_S**2 + i*n_S + i] = 1.0
        
        print('Begin optimization', flush=True)

        optLags, optgrad, optdual, optobj = dualoptFunc(n_S, Lags, Plist, UPlist, include)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj) #print-out function specific to the problem

        try:
            projLagnum = 2*n_S**2
            subLags = optLags[projLagnum:] #just split the multipliers corresponding to local constraints
            subLags, subproj_ulco_list, subproj_brco_list = split_generalP_rect_domains(subLags, subproj_ulco_list, subproj_brco_list, n_S=n_S)
            Lags = np.zeros(len(subLags)+projLagnum)
            Lags[:projLagnum] = optLags[:projLagnum]
            Lags[projLagnum:] = subLags[:] #always keep a copy of the global constraints
        except ValueError:
            print('subregions down to pixel level, iterations complete')
            break
    return 0


def dualopt_Msparse_iterative_splitting(n_S, Mx, My, chi, Ginv, dualoptFunc, outputFunc=None, pol='TM'):
    """
    runs dual optimization while iteratively increasing the number of projection constraints, seeding later optimizations with solution of prior optimizations
    dualoptfunc(n_S, initLags, GinvconjPlist, UPlist, include) returns dual optimum, optimizing from initLags based on Msparse formulation
    """

    design_ulco = (0,0)
    design_brco = (Mx,My)
    subproj_ulco_list = [(0,0)]
    subproj_brco_list = [(Mx,My)] #start with just the global constraints
    num_iters = 0
    while True:
        num_iters += 1
        proj_ulco_list = [design_ulco] + subproj_ulco_list
        proj_brco_list = [design_brco] + subproj_brco_list

        Plist = get_rect_Ideslist(design_ulco, design_brco, proj_ulco_list, proj_brco_list, sparse=True, pol=pol)
        num_region = len(Plist)
        print('Total # of subregions', num_region-1, flush=True)
        GinvconjPlist = []
        UPlist = []
        for i in range(num_region):
            GinvconjP = Ginv.conj().T @ Plist[i]
            UMP = (Ginv.conj().T @ GinvconjP.conj().T)/np.conj(chi) - GinvconjP.conj().T
            GinvconjPlist.append(GinvconjP)
            UPlist.append(UMP)

        #####SET THE INCLUDE LIST#####
        include = np.array([True] * (num_region * 2 * n_S**2))
        include[2*n_S**2:4*n_S**2] = False #since we include global constraints, leave out first subregion to prevent linear dependence

        if num_iters==1: #start of iteration, initialize Lags so only global Asym constraints in each source have non-zero multipliers
            Lags = np.zeros(len(include))
            for i in range(n_S):
                Lags[n_S**2 + i*n_S + i] = 1.0
        
        print('Begin optimization', flush=True)

        optLags, optgrad, optdual, optobj = dualoptFunc(n_S, Lags, GinvconjPlist, UPlist, include)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj) #print-out function specific to the problem

        try:
            projLagnum = 2*n_S**2
            subLags = optLags[projLagnum:] #just split the multipliers corresponding to local constraints
            subLags, subproj_ulco_list, subproj_brco_list = split_generalP_rect_domains(subLags, subproj_ulco_list, subproj_brco_list, n_S=n_S)
            Lags = np.zeros(len(subLags)+projLagnum)
            Lags[:projLagnum] = optLags[:projLagnum]
            Lags[projLagnum:] = subLags[:] #always keep a copy of the global constraints
        except ValueError:
            print('subregions down to pixel level, iterations complete')
            break

    return 0

def dualopt_Msparse_iterative_splitting_vacuum(n_S, Mx, My, design_mask, chi, Ginv, dualoptFunc, outputFunc=None, pol='TM'):
    """
    runs dual optimization while iteratively increasing the number of projection constraints, seeding later optimizations with solution of prior optimizations
    dualoptfunc(n_S, initLags, GinvconjPlist, UPlist, include) returns dual optimum, optimizing from initLags based on Msparse formulation
    """

    design_ulco = (0,0)
    design_brco = (Mx,My)
    subproj_ulco_list = [(0,0)]
    subproj_brco_list = [(Mx,My)] #start with just the global constraints
    num_iters = 0
    while True:
        num_iters += 1
        proj_ulco_list = [design_ulco] + subproj_ulco_list
        proj_brco_list = [design_brco] + subproj_brco_list

        Plist = get_rect_Ideslist_vacuum(design_mask, design_ulco, design_brco, proj_ulco_list, proj_brco_list, sparse=True, pol=pol)
        num_region = len(Plist)
        print('Total # of subregions', num_region-1, flush=True)
        GinvconjPlist = []
        UPlist = []
        temp_num_region = num_region
        for i in range(num_region):
            if len(np.nonzero(Plist[i])) == 0:
                temp_num_region -= 1
                continue
            GinvconjP = Ginv.conj().T @ Plist[i]
            UMP = (Ginv.conj().T @ GinvconjP.conj().T)/np.conj(chi) - GinvconjP.conj().T
            GinvconjPlist.append(GinvconjP)
            UPlist.append(UMP)
        num_region =  temp_num_region
        #####SET THE INCLUDE LIST#####
        include = np.array([True] * (num_region * 2 * n_S**2))
        include[2*n_S**2:4*n_S**2] = False #since we include global constraints, leave out first subregion to prevent linear dependence

        if num_iters==1: #start of iteration, initialize Lags randomly
            Lags = np.zeros(len(include))
            Lags[include] = np.random.rand(int(np.sum(include)))
        
        print('Begin optimization', flush=True)

        optLags, optgrad, optdual, optobj = dualoptFunc(n_S, Lags, GinvconjPlist, UPlist, include)

        if not (outputFunc is None):
            outputFunc(optLags, optgrad, optdual, optobj) #print-out function specific to the problem

        try:
            projLagnum = 2*n_S**2
            subLags = optLags[projLagnum:] #just split the multipliers corresponding to local constraints
            subLags, subproj_ulco_list, subproj_brco_list = split_generalP_rect_domains(subLags, subproj_ulco_list, subproj_brco_list, n_S=n_S)
            Lags = np.zeros(len(subLags)+projLagnum)
            Lags[:projLagnum] = optLags[:projLagnum]
            Lags[projLagnum:] = subLags[:] #always keep a copy of the global constraints
        except ValueError:
            print('subregions down to pixel level, iterations complete')
            break

    return 0

