#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 15:10:25 2019

@author: pengning
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.special as sp
from scipy import integrate
import mpmath
import os
axisfont = {'fontsize':'18'}


from .spherical_domain import *
from .dipole_field import *
    
#implement function that calculates coordinates of a field in spherical wave basis
#note that scipy sph_harm are the same convention as that of Kristensson
#except the theta and phi are flipped; scipy theta is azimuthal angle

import time

def check_lpmv_derivative(l,m,theta):
    x = np.cos(theta)
    sx = np.sin(theta)
    t0 = time.time()
    mycalc = 0.5*((l+m)*(l-m+1)*sp.lpmv(m-1,l,x)-sp.lpmv(m+1,l,x))
    t1 = time.time()
    [tmp,tmpdz] = sp.lpmn(m,l,x)
    libcalc = sx*tmpdz[m,l]
    t2 = time.time()
    print(mycalc)
    print('run time %.2f',t1-t0)
    print(libcalc)
    print('run time %.2f',t2-t1)
    
def check_sph_harm(l,m,theta,phi):
    print(sp.sph_harm(m,l,phi,theta))
    Clm = np.sqrt((2*l+1.0)/4/np.pi * sp.poch(l+abs(m)+1,-2*abs(m)))
    [Pmlv,Pmlvdiff] = sp.lpmn(abs(m),l,np.cos(theta))
    ans = Clm*Pmlv[abs(m),l]*np.exp(1j*m*phi)
    if (m<0):
        ans = (-1)**m * np.conjugate(ans)
    print(ans)
# routine that returns the vector values for vsph A1lm,A2lm,A3lm
def Cart_to_sphere(x,y,z):
    r = np.sqrt(x**2+y**2+z**2)
    phi = np.arctan2(y,x)
    if (r==0.0):
        theta = 0.0
    else:
        theta = np.arccos(z/r)
    
    
    sthe = np.sin(theta); cthe = np.cos(theta)
    sphi = np.sin(phi); cphi = np.cos(phi)
    rhat = np.array([sthe*cphi,sthe*sphi,cthe])
    thehat = np.array([cthe*cphi,cthe*sphi,-sthe])
    phihat = np.array([-sphi,cphi,0.0])
    
    return r,theta,phi,rhat,thehat,phihat

def get_vecsph(x,y,z,l,m):
    m_abs = abs(m) #calculate everything with |m|, then at the end conjugate and multiply (-1)^m if needed
    A1lm = np.zeros(3,dtype=complex); A2lm = np.zeros(3,dtype=complex); A3lm = np.zeros(3,dtype=complex)
    if m_abs>l:
        print ("m>l encountered in get_vecsph")
        return A1lm,A2lm,A3lm
    
    r,theta,phi,rhat,thehat,phihat = Cart_to_sphere(x,y,z)
    costheta = np.cos(theta)
    [Pmlv,Pmlvdiff] = sp.lpmn(m_abs,l,costheta) #assembling spherical harmonics by hand since we also need their derivatives
    Clm = np.sqrt((2*l+1.0)/4/np.pi * sp.poch(l+m_abs+1,-2*m_abs)) #prefactor for spherical harmonics
    phiphase = np.exp(1j*m_abs*phi)
    #A3lm = rhat*sp.sph_harm(m,l,phi,theta) #scipy harmonics have azimuthal angle as first angle argument
    A3lm = rhat * Clm*Pmlv[m_abs,l]*phiphase
    
    if theta==0.0 or costheta==-1.0: #special case, avoid 1/sin(theta) nan, see Kristensson appendix D
        if m_abs==1:
            prefact = np.sqrt((2*l+1.0)/16/np.pi)
            A1lm = -prefact*np.array([1j,-1.0,0.0])
            A2lm = -prefact*np.array([1.0,1j,0.0])
            if costheta==-1.0:
                A1lm = (-1)**l*A1lm; A2lm = (-1)**l*A2lm
        #otherwise nothing is done to A1lm, A2lm, and we get back 0s
    elif l>0: #A100 A200 are all 0
        pdvYphi_over_sine = 1j*m_abs*Clm*Pmlv[m_abs,l]*phiphase / np.sin(theta)
        pdvYthe = -np.sin(theta)*Clm*phiphase
        pdvYthe *= Pmlvdiff[m_abs,l]
        A1lm = (thehat*pdvYphi_over_sine - phihat*pdvYthe) / np.sqrt(l*(l+1.0))
        A2lm = (thehat*pdvYthe + phihat*pdvYphi_over_sine) / np.sqrt(l*(l+1.0))
        
    if m<0:
        A1lm = (-1)**m*np.conjugate(A1lm); A2lm = (-1)**m*np.conjugate(A2lm); A3lm = (-1)**m*np.conjugate(A3lm)
    return A1lm,A2lm,A3lm

#routine that returns the divergenceless regular spherical waves RgMe,o and RgNe,o

def get_rgwaves(k,x,y,z,l,m):
    rgMe = np.zeros(3); rgMo = np.zeros(3)
    rgNe = np.zeros(3); rgNo = np.zeros(3)
    if (l<1):
        print("l<1 encountered in get_regwave")
        return rgMe,rgMo,rgNe,rgNo
    
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    A1lm,A2lm,A3lm = get_vecsph(x,y,z,l,m)
    rgM = sp.spherical_jn(l,kr) * A1lm
    rgMe = np.real(rgM); rgMo = np.imag(rgM)
    jlprime = sp.spherical_jn(l,kr,derivative=True); jl_over_kr = sp.spherical_jn(l,kr)/kr
    rgN = (jlprime+jl_over_kr)*A2lm + np.sqrt(l*(l+1))*jl_over_kr*A3lm
    rgNe = np.real(rgN); rgNo = np.imag(rgN)
    
    return rgMe,rgMo,rgNe,rgNo

def get_rgM(k,x,y,z,l,m,par):
    if (l<1):
        print("l<1 encountered in get_regwave")
        return np.zeros(3)
    
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    A1lm,A2lm,A3lm = get_vecsph(x,y,z,l,m)
    rgM = sp.spherical_jn(l,kr) * A1lm
    if par%2==0:
        return np.real(rgM)
    else:
        return np.imag(rgM)
    
def get_rgN(k,x,y,z,l,m,par):
    if (l<1):
        print("l<1 encountered in get_regwave")
        return np.zeros(3)
    
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    A1lm,A2lm,A3lm = get_vecsph(x,y,z,l,m)
    jlprime = sp.spherical_jn(l,kr,derivative=True); 
    if (kr>0.0):
        jl_over_kr = sp.spherical_jn(l,kr)/kr
    elif (l==1):
        jl_over_kr = 1.0/3.0
    else:
        jl_over_kr = 0.0
    
    #print(jlprime)
    #print(jl_over_kr)
    rgN = (jlprime+jl_over_kr)*A2lm + np.sqrt(l*(l+1))*jl_over_kr*A3lm
    if par%2==0:
        return np.real(rgN)
    else:
        return np.imag(rgN)
    
def get_rgM_sqr(k,x,y,z,l,m,par):
    vec = get_rgM(k,x,y,z,l,m,par)
    return (np.vdot(vec,vec))

def get_rgwaves_sqr(k,x,y,z,l,m):
    rgMe,rgMo,rgNe,rgNo = get_rgwaves(k,x,y,z,l,m)
    return np.array([np.vdot(rgMe,rgMe),np.vdot(rgMo,rgMo),np.vdot(rgNe,rgNe),np.vdot(rgNo,rgNo)])

#implement a ball (centered at origin) integration function based on scipy tplquad
#func(x,y,z), but tplquad takes in (z,y,x)
def ball_tplquad(R,func):
    ylbound = lambda x: -np.sqrt(R**2-x**2)
    yhbound = lambda x: np.sqrt(R**2-x**2)
    zlbound = lambda x,y: -np.sqrt(R**2-x**2-y**2)
    zhbound = lambda x,y: np.sqrt(R**2-x**2-y**2)
    #e_abs = 1.5e-7
    #e_rel = 1e-4
    e_abs = 1.5e-8
    e_rel = 1.5e-8
    return integrate.tplquad(lambda z,y,x: func(x,y,z), -R,R, ylbound,yhbound, zlbound,zhbound, epsabs=e_abs,epsrel=e_rel)

def get_field_sqr(x,y,z,field_func):
    vec = field_func(x,y,z)
    return np.real(np.vdot(vec,vec))

def get_field_normsqr(R,field):
    ans,err = ball_tplquad(R,lambda x,y,z: get_field_sqr(x,y,z,field))
    return ans

def check_regwaves_normalization(k,R,l,m):
    #TODO vectorize the previous code to allow for quadpy parallel quadrature
    #scheme = quadpy.ball.hammer_stroud_14_3()
    #num_ans = scheme.integrate(lambda co: get_rgwaves_sqr(k,co[0],co[1],co[2],l,m), [0.0, 0.0, 0.0], r)
    #print(num_ans)
    #print(ball_tplquad(r,lambda x,y,z: get_rgM_sqr(k,x,y,z,l,m,0)))
    #print(ball_tplquad(r,lambda x,y,z: get_rgM_sqr(k,x,y,z,l,m,1)))
    print(get_field_normsqr(R,lambda x,y,z: get_rgM(k,x,y,z,l,m,0)))
    print(get_field_normsqr(R,lambda x,y,z: get_rgN(k,x,y,z,l,m,0)))
    rhoM = rho_M(l,k*R); rhoN = rho_N(l,k*R)
    print(1.0/k**3/(2-(m==0)) * rhoM)
    print(1.0/k**3/(2-(m==0)) * rhoN)
    
def get_regwave_coeff(k,R,norm,l,m,MorN,par,field):
    if (MorN=='M'):
        print('M')
        Re,err = ball_tplquad(R,lambda x,y,z: np.dot(get_rgM(k,x,y,z,l,m,par)/norm,np.real(field(x,y,z))))
        print("real rel. err:",err/abs(Re))
        Im,err = ball_tplquad(R,lambda x,y,z: np.dot(get_rgM(k,x,y,z,l,m,par)/norm,np.imag(field(x,y,z))))
        print("imag rel. err:",err/abs(Im))
    else:
        print('N')
        Re,err = ball_tplquad(R,lambda x,y,z: np.dot(get_rgN(k,x,y,z,l,m,par)/norm,np.real(field(x,y,z))))
        print("real rel. err:",err/abs(Re))
        Im,err = ball_tplquad(R,lambda x,y,z: np.dot(get_rgN(k,x,y,z,l,m,par)/norm,np.imag(field(x,y,z))))
        print("imag rel. err:",err/abs(Re))
    return Re+1j*Im

def get_coeffs_and_rhos_for_field(k,R,field):
    fnormsqr = get_field_normsqr(R,field)
    print(fnormsqr)
    cnormsqr = 0.0
    tolnormsqr = 0.01
    l=0
    rholist = []
    coefflist = []
    while (fnormsqr-cnormsqr)/fnormsqr > tolnormsqr:
        l += 1
        rhoM = rho_M(l,k*R); rhoN = rho_N(l,k*R)
        sqrtrhoM = np.sqrt(rhoM); sqrtrhoN = np.sqrt(rhoN)
        for m in range(l+1):
            print(l,',',m)
            prefact = np.sqrt(1.0/(2-(m==0))/k**3)
            
            cMe = get_regwave_coeff(k,R,prefact*sqrtrhoM,l,m,'M',0,field)
            rholist.append(rhoM); coefflist.append(cMe)
            tmp = np.real(np.conjugate(cMe)*cMe)
            cnormsqr += tmp
            print('Me normsqr: ',tmp)
            
            cMo = get_regwave_coeff(k,R,prefact*sqrtrhoM,l,m,'M',1,field)
            rholist.append(rhoM); coefflist.append(cMo)
            tmp = np.real(np.conjugate(cMo)*cMo)
            cnormsqr += tmp
            print('Mo normsqr: ',tmp)
            
            cNe = get_regwave_coeff(k,R,prefact*sqrtrhoN,l,m,'N',0,field)
            rholist.append(rhoN); coefflist.append(cNe)
            tmp = np.real(np.conjugate(cNe)*cNe)
            cnormsqr += tmp
            print('Ne normsqr: ',tmp)
            
            cNo = get_regwave_coeff(k,R,prefact*sqrtrhoN,l,m,'N',1,field)
            rholist.append(rhoN); coefflist.append(cNo)
            tmp = np.real(np.conjugate(cNo)*cNo)
            cnormsqr += tmp
            print('No normsqr: ',tmp)
    
    return np.array(coefflist), np.array(rholist)

#for a zpolarized dipole right underneath sphere, exploit cylindrical symmetry
#only non-zero coeff. are the RgNe_l0 coefficients

def get_coeffs_and_rhos_for_zdipole_field(k,R,zcoord,cofname):
    cof = open(cofname,'a')
    cof.write(str(zcoord))

    field = lambda x,y,z: zdipole_field(k,0,0,zcoord,x,y,z)
    fnormsqr = get_field_normsqr(R,field)
    print(fnormsqr)
    cnormsqr = 0.0
    tolnormsqr = 0.001
    l=0
    rholist = []
    coefflist = []
    while (fnormsqr-cnormsqr)/fnormsqr > tolnormsqr:
        l += 1
        rhoN = rho_N(l,k*R)
        sqrtrhoN = np.sqrt(rhoN)
        m = 0 #just m=0
        print(l,',',m)
        prefact = np.sqrt(1.0/k**3)

        cNe = get_regwave_coeff(k,R,prefact*sqrtrhoN,l,m,'N',0,field)
        cof.write(" "+str(cNe))
        cof.flush()
        os.fsync(cof)
        
        rholist.append(rhoN); coefflist.append(cNe)
        tmp = np.real(np.conjugate(cNe)*cNe)
        cnormsqr += tmp
        print('Ne normsqr: ',tmp)
    
    cof.write("\n")
    cof.close()        
    return np.array(coefflist), np.array(rholist)
#implement single source LM formulation for scattering/absorption

def lagrange_mul(lbda,zeta,csqrs,rhos):
    tmp = 1.0 + (1.0-zeta*rhos)/(1.0+lbda*zeta*rhos)
    return np.sum(csqrs*tmp/(1.0+lbda*zeta*rhos))

def get_lagrange_mul(zeta,csqr,rhos):
    if lagrange_mul(0.0,zeta,csqr,rhos)>=0:
        return 0.0
    
    lbdamin = 0.0; lbdamax = 1.0
    while lagrange_mul(lbdamax,zeta,csqr,rhos)<0:
        lbdamax*=2.0
    
    while (lbdamax-lbdamin)/lbdamax > 1e-4:
        lbdamid = (lbdamin+lbdamax)/2.0
        if lagrange_mul(lbdamid,zeta,csqr,rhos)<0:
            lbdamin = lbdamid
        else:
            lbdamax = lbdamid
    
    return (lbdamin+lbdamax)/2.0

def Psca_bound_zdipole(k,zeta,lbda,csqrs,rhos):
    tmp = csqrs*(1.0-lbda+2.0*zeta*lbda*rhos)/4/(1.0+lbda*zeta*rhos)**2
    return 0.5*k*zeta*(1.0+lbda)*np.sum(tmp)


def test_write(fname,string):
    file = open(fname,'a')
    file.write(string)
    file.flush()
    os.sync
    file.write(np.array([1,2,3]))
    file.close()


def Psca_bound_zeta0246_to_file(k,R,zdist,fname='k1R1_zeta0246_dist_Psca.txt',cofname='k1R1_zdipole_dist_coeffs.txt'):
    file = open(fname,'a')
    
    zetas = 10.0**np.array([0,2,4,6])
    zcoord = -R-zdist
    cs,rhos = get_coeffs_and_rhos_for_zdipole_field(k,R,zcoord,cofname)
    csqrs = np.real(np.conjugate(cs)*cs)
    
    
    file.write(str(zcoord))
    for i in range(len(zetas)):
        zeta = zetas[i]
        lbda = get_lagrange_mul(zeta,csqrs,rhos)
        Psca = Psca_bound_zdipole(k,zeta,lbda,csqrs,rhos)
        file.write(" ")
        file.write(str(Psca))
    
    file.write("\n")
    file.close()
    
def Psca_bound_zdistlist_zeta0246_to_file(k,R,zdistlist,fname='k1R1_zeta0246_dist_Psca.txt',cofname='k1R1_zdipole_dist_coeffs.txt'):
    for i in range(len(zdistlist)):
        print(zdistlist[i])
        Psca_bound_zeta0246_to_file(k,R,zdistlist[i],fname,cofname)
        
def plot_zdipole_Psca_bound_zeta0246(fname='k1R1_zeta0246_dist_Psca.txt'):
    zetas = 10.0**np.array([0,2,4,6])
    tmp = np.loadtxt(fname)
    R=1.0; k=1.0 #change this later maybe
    zdist = np.abs(tmp[:,0]) - 1.0
    zorder = np.argsort(zdist)
    
    print((np.log(tmp[zorder[-1],2])-np.log(tmp[zorder[0],2]))/(np.log(zdist[zorder[-1]])-np.log(zdist[zorder[0]])))
    plt.figure()
    for i in range(len(zetas)):
        zeta = zetas[i]
        plt.loglog(zdist[zorder]*k/2.0/np.pi,tmp[zorder,i+1],label='$\zeta=$'+'{:0.0E}'.format(zeta))
        
    plt.xlabel('$d/\lambda$',**axisfont)
    plt.ylabel('bound for $P_{sca}$ (a.u.)',**axisfont)

    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('dipole_Psca.png',dpi=400)
    plt.show()


####an implementation based on analytica formulae in Kardar RHT paper###

def get_outN(k,x,y,z,l,m):
    if (l<1):
        print("l<1 encountered in get_outwave")
        return np.zeros(3)
    
    r = np.sqrt(x**2+y**2+z**2)
    kr = k*r
    A1lm,A2lm,A3lm = get_vecsph(x,y,z,l,m)
    jlprime = sp.spherical_jn(l,kr,derivative=True); 
    if (kr>0.0):
        jl_over_kr = sp.spherical_jn(l,kr)/kr
    elif (l==1):
        jl_over_kr = 1.0/3.0
    else:
        jl_over_kr = 0.0
    
    ylprime = sp.spherical_yn(l,kr,derivative=True)
    yl_over_kr = sp.spherical_yn(l,kr) / kr
    #print(jlprime)
    #print(jl_over_kr)
    outN = (jlprime+1j*ylprime + jl_over_kr+1j*yl_over_kr)*A2lm + np.sqrt(l*(l+1))*(jl_over_kr+1j*yl_over_kr)*A3lm
    return outN

# the below is consistent with zdipole_field()
def zdipole_field_from_spherical_wave(k,xd,yd,zd,xp,yp,zp):
    return 1j*k*np.sqrt(1.0/6/np.pi) * get_outN(k,xp-xd,yp-yd,zp-zd,1,0)

def xdipole_field_from_spherical_wave(k,xd,yd,zd,xp,yp,zp):
    prefact = -1j*k/np.sqrt(12*np.pi)
    return prefact*(get_outN(k,xp-xd,yp-yd,zp-zd,1,1) - get_outN(k,xp-xd,yp-yd,zp-zd,1,-1))
#for zpolarized dipole along z axis only the RgNl0 waves are needed to represent dipole field
#for region above dipole should use U^-, A^- in Kardar paper expression
    
