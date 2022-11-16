#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 11:01:10 2020

@author: pengning
"""
import numpy as np
import mpmath
from mpmath import mp

def mp_normsqr(cplx):
    return mp.re(mp.conj(cplx)*cplx)

def mp_conjdot(vec1,vec2):
    #does <v1|v2>
    return (vec1.transpose_conj()*vec2)[0,0]

def mp_vecnormsqr(vec):
    return mp.re((vec.transpose_conj()*vec)[0,0])

def mp_dblabsdot(vec1,vec2):
    a1 = np.squeeze(np.abs(vec1.tolist()))
    a2 = np.squeeze(np.abs(vec2.tolist()))
    return np.dot(a1,a2)

def mp_Lsolve(L,b):
    n = L.rows
    x = b.copy()
    for i in range(n):
        x[i] -= mp.fsum(L[i,j]*x[j] for j in range(i))
        x[i] /= L[i,i]
    return x

def mp_CholeskyLsolve(L,b):
    x = mp_Lsolve(L,b)
    x = mp.U_solve(L.H, x)
    return x

def form_square_block_matrix(mat1,mat2):
    """
    forms an amalgamated square block matrix out of square matrices mat1 and mat2
    if mat1 is mxm and mat2 is nxn, the returned matrix is (m+n)x(m+n) with mat1 and mat2 on the block diagonal
    if mat1 is mx1 and mat2 is nx1, the returned vector is (m+n)x1 with mat1 and mat2 stacked together
    """
    if mat1.cols==1:
        mat3 = mp.matrix(mat1.rows+mat2.rows,1)
        mat3[:mat1.rows] = mat1[:]
        mat3[mat1.rows:mat3.rows] = mat2[:]
    else:
        mat3 = mp.matrix(mat1.rows+mat2.rows, mat1.rows+mat2.rows)
        mat3[:mat1.rows,:mat1.rows] = mat1[:,:]
        mat3[mat1.rows:mat3.rows,mat1.rows:mat3.rows] = mat2[:,:]
    return mat3
