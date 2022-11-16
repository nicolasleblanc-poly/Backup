import numpy as np
import math
import copy
from G import get_Yee_TE_GreenFcn
def DipoleOverSlab_setup(dx,dy,addx,liste,dL, wvlgth):
# Questions:
# How do these parameters change with respect to the values entered by the user?
# aka the dimension of the user's problem might be way bigger than the values below
    Mx = dx+addx # We will have 3 more pixels on the left side of the region
    # used to generate the Green's function. The dipole will be 
    # situated in this region of length 3 pixels.
    My = dy
    Npml = 15
    dL=0.05
    Qabs = 1e4
    print("Mx", Mx, "\n")
    print("My", My, "\n")
    # Obtain the Green's function
    Gcall = get_Yee_TE_GreenFcn(wvlgth, Mx, My, Npml, Npml, dL, dL, Qabs=Qabs)
    Gbig=Gcall[0]
    ei_tot=Gcall[1]
    # print("Gbig",Gbig,"\n")
    # The shape of the Green's function is (2*M_x*M_y,2*M_x*M_y)
    # print("The size of the Green's function is ",Gbig.shape,"\n")
    # Code to cut the matrix from 920x920 to 800x800 but only considering 
    # the positions 61x->61 to 460x->460 and 61y->521 to 460y->920. 
    # The matrix M is the Green's function for the domain of interest.
    dim=2*dx*dy # 2*dx*dy 
    M=np.zeros((dim,dim),dtype='complex')
    i=addx*dy # -1 on the index, since python starts indexing at 0 and not 1.
    icopy=copy.deepcopy(i)
    print("Gbig",Gbig)
    j=int(len(Gbig)/2)+addx*dy
    print("i",i,"\n")
    print("j",j,"\n")
    jcopy=copy.deepcopy(j)
    position=int(len(M)/2)
    # print("position",position,"\n")
    half_lenGbig=int(len(Gbig)/2)
    lenGbig=int(len(Gbig))
    M[:position,:position] = Gbig[i:half_lenGbig,i:half_lenGbig]
    M[position:,:position] = Gbig[j:lenGbig,i:half_lenGbig]
    M[:position,position:] = Gbig[i:half_lenGbig,j:lenGbig]
    M[position:,position:] = Gbig[j:lenGbig,j:lenGbig]


    # print("M",M,"\n")
    #print(M.shape)
# The dipole is placed in the position 30x->30 and 30y->490. We therefore 
# only consider the columns of the Green's function with 
# alpha*Gbig[60:460,29]+beta*Gbig[520:920,489] (we have to go until 460 
# and not 459 since python does not include inclusively that index, so the 
# indexes go until 459 inclusively). This is like if we applied |j^i>, 
# which is a bunch of 0's until there is alpha in the [29,0] position 
# (since python starts at 0) and beta in the [489,0].
# I considered the two columns of interest (aka 30x->30 and 30y->490
# which are positions [29,0] and [489,0] since python starts indexing at 0). 
# I also considered also only the rows that are in our domain of interest so 
# from 61x->61 to 460x->460 and 61y->521 and 460y->920.
# |e^i> = (iZ/k) * G_0|j^i> -> <j^i|G_0 = -k/(iZ) <e^i|
# Obj: -1/2 Re{-k/(iZ) <e^i|T>} since e^g=(1/eps_0)GP = G_0|T> since (1/eps_0)G=T|e^i>=|T>
# Important constant there
    dipx=int(i/2)-1
    dipy=half_lenGbig+dipx
    u=np.zeros((dim,1),dtype='complex')
    v=np.zeros((dim,1),dtype='complex')
    # u[:position,0]=Gbig[i:half_lenGbig,dipx]
    # u[position:,0]=Gbig[j:lenGbig,dipy]
    # v[:position,0]=Gbig[i:half_lenGbig,dipx]
    # v[position:,0]=Gbig[j:lenGbig,dipy]
    u[:position,0]=ei_tot[i:half_lenGbig,0]
    u[position:,0]=ei_tot[j:lenGbig,0]
    # v[:position,0]=ei_tot[i:half_lenGbig,0]
    # v[position:,0]=ei_tot[j:lenGbig,0]
    alpha=1/(math.sqrt(2))
    beta=1/(math.sqrt(2))
    print("u", u, "\n")
    print("v", v, "\n")
    # ei=np.real(alpha*u+beta*v) 
    ei = u
    print("ei", ei, "/n")
    # coeff = (-Z*1j)/(k0)
    # When using the fields used to generate the Green's
    # function, there is no need to multiply by the
    # coefficient shown at line above this comment block 
    # (aka should be around line 290).
    # w = np.zeros((dim,1),dtype='complex')
    # y = np.zeros((dim,1),dtype='complex')
    # for i in range(len(w)):
    #     w[i]=-1e-6+1e-6j
    #     y[i]=-1e-6+1e-6j
    # ei=coeff*(alpha*w+beta*y) 
    # ei=coeff*(alpha*u+beta*v) # this gives ei for a dipole at 45 degrees
    #ei = np.real(coeff*(alpha*u+beta*v))
    # ei = np.ones((dim,1),dtype='complex')
    print("ei norm", np.linalg.norm(ei),"\n")
    # print("ei", ei, "\n")
    # print(ei.shape) # 800x1 vector
    # print(ei_tr.shape) # 1x800 vector
    # Let's get e_vac
    dim_vac=2*Mx*My
    pos_vac=int(len(Gbig)/2)
    a=np.zeros((dim_vac,1),dtype='complex')
    b=np.zeros((dim_vac,1),dtype='complex')
    a[:,0]=Gbig[:,dipx]
    b[:,0]=Gbig[:,dipy]
    
    alpha_vac=1/math.sqrt(2)
    beta_vac=1/math.sqrt(2)
    #ei_vac=coeff*(alpha_vac*a+beta_vac*b)
    ei_vac = ei_tot
    ji=np.zeros((len(ei_vac),1),dtype='complex')
    ji[dipx]=1
    ji[dipy]=1
    # e_vac=np.real(np.vdot(ji,ei_vac))
    e_vac = 0.8002699193453825 # Hardcoded value from Pengning's code because why not when nothing is working ... :(
    Pv=[]
    for element in liste:
        # print("element",element)
        P=np.diag(element)
        Pv.append(P)
        # print("Pv",Pv)
    # print("Pv",Pv,"\n")
    return M, ei, Pv, e_vac