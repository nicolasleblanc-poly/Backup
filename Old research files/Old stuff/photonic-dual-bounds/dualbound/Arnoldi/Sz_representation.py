#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 10:07:48 2022

@author: pengning
"""

import numpy as np
import py3nj #keep in mind py3nj works with integer arguments equal to double the symbol arguments
import scipy.linalg as la


def get_Sz_representation_TJ(M, J_max):
    """
    get representation of Sz in a vector spherical harmonics basis A^{T}_JM
    M is fixed, since Sz does not mix M numbers
    the TJ index is organized in the order (T=1,J=M) (T=2,J=M) (T=3,J=M) (T=1,J=M+1)...
    if necessary can probably tighten up repeated evaluation of wigner symbols later
    """
    M_abs = abs(M)
    if J_max<M_abs:
        raise ValueError('J_max less than abs(M), unphysical')
    
    basis_size = 3*(J_max-M_abs+1)
    Sz_mat = np.zeros((basis_size,basis_size), dtype=np.complex)
    S = 1 #photon spin=1
    for l in range(J_max-M_abs+1):
        J = l+M_abs
        
        #indices in basis for relevant VSH
        ind_1J = 3*l
        ind_2J = 3*l+1
        ind_3J = 3*l+2
        
        ind_1Js1 = 3*(l-1)
        ind_2Js1 = 3*(l-1)+1
        ind_3Js1 = 3*(l-1)+2
        
        ind_1Jp1 = 3*(l+1)
        ind_2Jp1 = 3*(l+1)+1
        ind_3Jp1 = 3*(l+1)+2
        
        global_sqrt_fac = np.sqrt((2*J+1) * S * (S+1) * (2*S+1))
        
        ########get Sz action on A^{3}_{J,M}########
        #doing A^{3} first since in the case J=0, A^{1} and A^{2} don't exist
        if J<J_max:
            phase = -1j #(-1)^(2J+1) * 1j
            wigfactor = (py3nj.wigner6j(2*J, 2*(J+1), 2  ,  2, 2, 2*(J+1))
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J+1)  ,  2*M, 0, 2*M))
            #A^{1}_{J+1,M} contribution
            Sz_mat[ind_1Jp1, ind_3J] += (global_sqrt_fac * phase * wigfactor
                                         * np.sqrt((J+1)/(2*J+1)))
        
        if J==0:
            continue #for J==0, A^{2} and A^{3} don't exist, A^{3}_00 is coupled just to A^{1}_{10}
        
        if l>0:
            phase = 1j #(-1)*(2J) * 1j
            wigfactor = (py3nj.wigner6j(2*J, 2*(J-1), 2  ,  2, 2, 2*(J-1))
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J-1)  ,  2*M, 0, 2*M))
            #A^{1}_{J-1,M} contribution
            Sz_mat[ind_1Js1, ind_3J] += (global_sqrt_fac * phase * wigfactor
                                        * np.sqrt(J/(2*J+1)))
        
        phase = -1 #(-1)^(2J+1)
        wigfactor = (py3nj.wigner6j(2*J, 2*J, 2  ,  2, 2, 2*(J-1))
                     * py3nj.clebsch_gordan(2*J, 2, 2*J  ,  2*M, 0, 2*M))
        #first part of A^{2}_{J,M} contribution
        Sz_mat[ind_2J, ind_3J] += (global_sqrt_fac * phase * wigfactor
                                   * np.sqrt(J*(J+1))/(2*J+1))
        #first part of A^{3}_{J,M} contribution
        Sz_mat[ind_3J, ind_3J] += (global_sqrt_fac * phase * wigfactor
                                   * J/(2*J+1))
        
        phase = 1 #(-1)^(2J+2)
        wigfactor = (py3nj.wigner6j(2*J, 2*J, 2  ,  2, 2, 2*(J+1))
                     * py3nj.clebsch_gordan(2*J, 2, 2*J  ,  2*M, 0, 2*M))
        #second part of A^{2}_{J,M} contribution
        Sz_mat[ind_2J, ind_3J] += (global_sqrt_fac * phase * wigfactor
                                   * np.sqrt(J*(J+1))/(2*J+1))
        #second part of A^{3}_{J,M} contribution, pay attention to extra - sign
        Sz_mat[ind_3J, ind_3J] += (-global_sqrt_fac * phase * wigfactor
                                   * (J+1)/(2*J+1))
        
        
        
        ########get Sz action on A^{1}_{J,M}########
        if l>0:
            phase = 1j #(-1j)*(-1)^{2J+1}
            wigfactor = (py3nj.wigner6j(2*J, 2*(J-1), 2  ,  2, 2, 2*J) 
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J-1)  ,  2*M, 0, 2*M))
            
            #A^{2}_{J-1,M} contribution
            Sz_mat[ind_2Js1 , ind_1J] += (global_sqrt_fac * phase * wigfactor 
                                         * np.sqrt((J-1)/(2*J-1)))
            #A^{3}_{J-1,M} contribution, pay attention to extra - sign
            Sz_mat[ind_3Js1 , ind_1J] += (-global_sqrt_fac * phase * wigfactor
                                         * np.sqrt(J/(2*J-1)))
        
        phase = 1 #-1j*(-1)^{2*J+2}*1j, just 1
        wigfactor = (py3nj.wigner6j(2*J, 2*J, 2  ,  2, 2, 2*J)
                     * py3nj.clebsch_gordan(2*J, 2, 2*J  ,  2*M, 0, 2*M))
        #A^{1}_{J,M} contribution
        Sz_mat[ind_1J, ind_1J] += global_sqrt_fac * phase * wigfactor
        
        if J<J_max:
            phase = 1j #-1j*(-1)^{2*J+1}
            wigfactor = (py3nj.wigner6j(2*J, 2*(J+1), 2  ,  2, 2, 2*J)
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J+1), 2*M, 0, 2*M))
            #A^{2}_{J+1,M} contribution
            Sz_mat[ind_2Jp1, ind_1J] += (global_sqrt_fac * phase * wigfactor
                                      * np.sqrt((J+2)/(2*J+3)))
            #A^{3}_{J+1,M} contribution
            Sz_mat[ind_3Jp1, ind_1J] += (global_sqrt_fac * phase * wigfactor
                                      * np.sqrt((J+1)/(2*J+3)))
        
        ########get Sz action on A^{2}_{J,M}########
        if l>0:
            phase = 1j #(-1)^{2J}*1j
            wigfactor = (py3nj.wigner6j(2*J, 2*(J-1), 2  ,  2, 2, 2*(J-1))
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J-1)  ,  2*M, 0, 2*M))
            #A^{1}_{J-1,M} contribution
            Sz_mat[ind_1Js1, ind_2J] += (global_sqrt_fac * phase * wigfactor
                                         * np.sqrt((J+1)/(2*J+1)))
        
        phase = -1 #(-1)^{2J+1}
        wigfactor = (py3nj.wigner6j(2*J, 2*J, 2  ,  2, 2, 2*(J-1))
                     * py3nj.clebsch_gordan(2*J, 2, 2*J  ,  2*M, 0, 2*M))
        #first part of A^{2}_{J,M} contribution
        Sz_mat[ind_2J, ind_2J] += (global_sqrt_fac * phase * wigfactor
                                   * (J+1)/(2*J+1))
        #first part of A^{3}_{J,M} contribution
        Sz_mat[ind_3J, ind_2J] += (global_sqrt_fac * phase * wigfactor
                                   * np.sqrt((J+1)*J)/(2*J+1))
        
        phase = -1 #(-1)^{2J+1}
        wigfactor = (py3nj.wigner6j(2*J, 2*J, 2  ,  2, 2, 2*(J+1))
                     * py3nj.clebsch_gordan(2*J, 2, 2*J  ,  2*M, 0, 2*M))
        #second part of A^{2}_{J,M} contribution
        Sz_mat[ind_2J, ind_2J] += (global_sqrt_fac * phase * wigfactor
                                   * J/(2*J+1))
        #second part of A^{3}_{J,M} contribution, pay attention to extra - sign
        Sz_mat[ind_3J, ind_2J] += (-global_sqrt_fac * phase * wigfactor
                                   * np.sqrt(J*(J+1))/(2*J+1))
        
        if J<J_max:
            phase = 1j #(-1)^(2*J) * 1j
            wigfactor = (py3nj.wigner6j(2*J, 2*(J+1), 2  ,  2, 2, 2*(J+1))
                         * py3nj.clebsch_gordan(2*J, 2, 2*(J+1)  ,  2*M, 0, 2*M))
            #A^{1}_{J+1,M} contribution
            Sz_mat[ind_1Jp1, ind_2J] += (global_sqrt_fac * phase * wigfactor
                                         * np.sqrt(J/(2*J+1)))
        
        
        
    return Sz_mat


def Sz_dot_angular_radial(Sz, ar):
    """
    convenience function that evaluates effect of Sz on angular radial representation
    assume that ar and Sz have the same 0-th dimension
    """
    Sz_ar = np.zeros_like(ar, dtype=np.complex)
    for i in range(Sz.shape[0]):
        Sz_ar += np.outer(Sz[:,i], ar[i,:])
    return Sz_ar


def vdot_angular_radial(ar1, ar2, rsqrgrid, rdiffgrid):
    """
    function that computes inner product between two vector fields in so-called angular radial representation
    ar1/2 is a 2D array, with first index going over angular dependence in terms of VSH A^{T}_JM
    and second index over rgrid. The rgrid dot product is handled similar to shell_Green_Taylor_Arnoldi_spatialDiscretization.py
    """
    r_vdot_integrand = np.conj(ar1)*ar2 * rsqrgrid #use numpy broadcasting
    return np.sum((r_vdot_integrand[:,:-1]+r_vdot_integrand[:,1:])*rdiffgrid/2.0)
    


def Arnoldi_inverse_converge(M, v0, invnormtol=1e-5, singulartol=1e-10, min_iter=None):
    """
    do Arnoldi iteration of matrix M starting with initial vector v0
    termination when norm of M^-1 v0 in Arnoldi basis converges to within relative invnormtol
    """
    if min_iter is None:
        min_iter = M.shape[0]
    
    v0 /= la.norm(v0) #normalize v0 if it isn't already
    vlist = [v0]
    vlist.append(M @ v0)
    A = np.zeros((1,1), dtype=np.complex)
    A[0,0] = np.vdot(v0, vlist[1])
    invnorm = np.abs(1.0 / A[0,0])
    #vlist will always be 1 longer than the size of A
    while True:
        print('iteration number', A.shape[0], 'invnorm', invnorm)
        if len(vlist)>M.shape[0]: #Arnoldi vector # exceending original operator dimension
            break
        A_new = np.zeros((A.shape[0]+1,A.shape[1]+1), dtype=np.complex)
        A_new[:-1,:-1] = A
        #first normalize image from previous iteration
        for i in range(len(vlist)-1):
            vlist[-1] -= A[i,-1] * vlist[i]
        A_new[-1, -2] = la.norm(vlist[-1]) #coefficient of vlist[-2] for vlist[-1] must be the normalization factor since all orthogonal vectors vlist[0..-3] have been taken out
        print('norm of new Arnoldi vector', A_new[-1,-2])
        if A_new[-1,-2] < singulartol: #if we have encountered a self-contained Krylov subspace
            break
        vlist[-1] /= A_new[-1,-2] #normalized
        
        vlist.append(M @ vlist[-1])
        for i in range(len(vlist)-1):
            A_new[i,-1] = np.vdot(vlist[i], vlist[-1])
        A = A_new

        """
        #check inverse norm
        b = np.zeros(A.shape[0])
        b[0] = 1.0
        tmp = la.norm(la.solve(A, b))
        if A.shape[0]>=min_iter and abs(tmp-invnorm)<invnorm * invnormtol:
            break
        invnorm = tmp
        
        #for debugging only
        if A.shape[0]==2:
            print('2x2 Arnoldi', A)
        """
        
    print('final invnorm', invnorm)
    print('shape of A', A.shape)
    return A, vlist[:-1]

        
    
    
