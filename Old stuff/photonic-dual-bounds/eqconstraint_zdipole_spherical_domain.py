#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 15:02:15 2019

@author: pengning
"""

import numpy as np
import mpmath
from mpmath import mp
from sympy.physics.wigner import wigner_3j
from dualbound.Arnoldi.spherical_domain import mp_rho_N
from dualbound.Arnoldi.dipole_field import mp_spherical_hn, zdipole_field
from dualbound.Arnoldi.dipole_sphere import zdipole_field_from_spherical_wave, get_rgN


######recurrence relation + tabulation for wigner3j symbols for efficiency
def wigner3j_recurrence(l,pm,wigner1llp1,wigner1llm1):
    mp.dps = mp.dps*2
    #routine for computing successive Wigner3j symbols for z-polarized dipole based on recurrence
    #wigner1llp1 stores wigner symbols of the form (1 l l+1 ; 0 0 0)
    #wigner1llm1 stores wigner symbols of the form (1 l l-1 ; 0 0 0)
    #recurrences found on the Wolfram Functions site
    if len(wigner1llp1)<2:
        wigner1llp1.clear(); wigner1llm1.clear()
        wigner1llp1.extend([mp.mpf(wigner_3j(1,1,2,0,0,0).evalf(mp.dps)),
                            mp.mpf(wigner_3j(1,2,3,0,0,0).evalf(mp.dps))])
        wigner1llm1.extend([mp.mpf(wigner_3j(1,1,0,0,0,0).evalf(mp.dps)),
                            mp.mpf(wigner_3j(1,2,1,0,0,0).evalf(mp.dps))])

    i = len(wigner1llp1)
    while i<l:
        i = i+1
        mp_i = mp.mpf(i)
        wigner1llm1.append(-mp.sqrt(mp_i*(2*mp_i-3)/((mp_i-1)*(2*mp_i+1))) * wigner1llp1[i-3])
        #wigner1llm1.append(-sp.sqrt(sp.Rational(i*(2*i-3),(i-1)*(2*i+1)))*wigner1llp1[i-3])
        #since python lists start with index 0, to get (1 i-2 i-1;0 0 0) need wigner1llp1[i-2-1]
        #wigner1llp1.append(-sp.sqrt(sp.Rational((2*i-1)*(i+1),i*(2*i+3)))*wigner1llm1[i-1])
        wigner1llp1.append(-mp.sqrt((2*mp_i-1)*(mp_i+1)/(mp_i*(2*mp_i+3))) * wigner1llm1[i-1])
    
    mp.dps = mp.dps//2
    if pm==1:
        return wigner1llp1[l-1]
    else:
        return wigner1llm1[l-1]


def mp_get_normalized_RgNl0_coeff_for_zdipole_field_recurrence(k,R,dist,l,rhoN_l,wigner1llp1,wigner1llm1):
    #includes sv and wigner3j symbols in argument
    ans = mpmath.mpc(0.0j)
    kd = k*(R+dist) #dist is distance between dipole and sphere surface, the translation distance d is between the two origins so dipole and sphere center
 #   norm = np.sqrt(rho_N(l,k*R)/k**3) #normalization
    mpimag = mp.mpc(1j)
    for nu in range(l-1,l+2):
        if nu==l:
            continue #the wigner symbol (1 l l;0 0 0) == 0
        tmp = mpimag**(1-nu-l)*0.5*(2+l*(l+1)-nu*(nu+1))*(2*nu+1)*mpmath.sqrt(3*(2*l+1)/2/l/(l+1))
        wignerfactor = wigner3j_recurrence(l,nu-l,wigner1llp1,wigner1llm1)**2
        #tmp *= mp.mpf(wignerfactor.evalf(mp.dps)) * mp_spherical_hn(nu,kd)
        tmp *= wignerfactor * mp_spherical_hn(nu,kd)
        ans += tmp
    #print(ans)
    ans *= 1j*k*mpmath.sqrt((1.0/6/mpmath.pi) * rhoN_l/k**3) #normalization included in sqrt
    return ans


def check_zdipole_spherical_expansion(k,R, xp,yp,zp, dist):
    print("original zdipole field function")
    print(zdipole_field(k,0,0,-R-dist,xp,yp,zp))
    print("spherical wave expansion dipole field")
    print(zdipole_field_from_spherical_wave(k,0,0,-R-dist,xp,yp,zp))
    
    wig_1llp1_000 = []; wig_1llm1_000 = [] #setup wigner3j lists
    
    cs = []
    cnormsqr = 0.0
    cnormsqr_old = 0.0
    l=0
    lmax = 20; delta_l = 10
    while True:
        l+=1
        rhoN = mp_rho_N(l,k*R)
        cl = mp_get_normalized_RgNl0_coeff_for_zdipole_field_recurrence(k,R,dist, l, rhoN, wig_1llp1_000,wig_1llm1_000)
        cs.append(cl)

        cnormsqr += mp.re( np.sum(np.conjugate(cl)*cl) )
#        print(cnormsqr)
        if l==lmax:
            print('l', l)
            print('cnormsqr', cnormsqr)
            if (cnormsqr-cnormsqr_old)/cnormsqr < 1e-4:
                break
            lmax += delta_l
            cnormsqr_old = cnormsqr
    print(l)
    
    expfield1 = np.array([mp.zero,mp.zero,mp.zero])
    for i in range(1,l+1):

        RgNe_field = get_rgN(k,xp,yp,zp, i,0,0)

        rhoN = mp_rho_N(i,k*R)
        normN = mp.sqrt(rhoN / k**3)
        expfield1 += RgNe_field*cs[i-1]/normN
        
    print("expansion field via real spherical waves is")
    print(expfield1)

