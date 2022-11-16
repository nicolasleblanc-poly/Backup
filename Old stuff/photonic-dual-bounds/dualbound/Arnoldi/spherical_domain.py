#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 15:43:25 2019

@author: pengning
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as sp
import mpmath
from mpmath import mp


def rho_M(l,kr): #singular value of Asym(G) associated with RgM
    return 0.25*np.pi*kr**2 * (sp.jv(l+0.5,kr)**2 - sp.jv(l-0.5,kr)*sp.jv(l+1.5,kr))

def rho_N(l,kr): #singular value of Asym(G) associated with RgN
    v1 = ((l+1.0)/(2.0*l+1.0)) * (sp.jv(l-0.5,kr)**2 - sp.jv(l+0.5,kr)*sp.jv(l-1.5,kr))
    v2 = (l/(2.0*l+1.0)) * (sp.jv(l+1.5,kr)**2 - sp.jv(l+0.5,kr)*sp.jv(l+2.5,kr))
    return 0.25*np.pi*kr**2 * (v1+v2)

#high precision (mpmath) routines
def mp_rho_M(l,kr):
    return 0.25*mp.pi*kr**2 * (mpmath.besselj(l+0.5,kr)**2 - 
                               mpmath.besselj(l-0.5,kr)*mpmath.besselj(l+1.5,kr))

def mp_rho_N(l,kr):
    if l==1 and kr==0.0:
        return 0.0
    v1 = ((l+1.0)/(2.0*l+1.0)) * (mpmath.besselj(l-0.5,kr)**2 - mpmath.besselj(l+0.5,kr)*mpmath.besselj(l-1.5,kr))
    v2 = (l/(2.0*l+1.0)) * (mpmath.besselj(l+1.5,kr)**2 - mpmath.besselj(l+0.5,kr)*mpmath.besselj(l+2.5,kr))
    return 0.25*mp.pi*kr**2 * (v1+v2)