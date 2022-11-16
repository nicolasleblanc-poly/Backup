#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 23 16:11:10 2019

@author: pengning
"""

import numpy as np
import matplotlib.pyplot as plt

import scipy.special as sp
import mpmath
from mpmath import mp

from . import spherical_domain as sph #shell domain sv can be computed as difference in sphere domain sv

def shell_rho_M(l,kR1,kR2): #kR1 inner radius kR2 outer radius
    return sph.rho_M(l,kR2) - sph.rho_M(l,kR1)

def shell_rho_N(l,kR1,kR2):
    return sph.rho_N(l,kR2) - sph.rho_N(l,kR1)

def mp_shell_rho_M(l,kR1,kR2):
    return sph.mp_rho_M(l,kR2) - sph.mp_rho_M(l,kR1)

def mp_shell_rho_N(l,kR1,kR2):
    return sph.mp_rho_N(l,kR2) - sph.mp_rho_N(l,kR1)