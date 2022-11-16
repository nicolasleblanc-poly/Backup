import numpy as np
from timeit import default_timer as timer
from LagrangeMultSolver import lsolverpick
from setup import DipoleOverSlab_setup
# Code for the optimization of the extracted power 
def extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv, e_vac):
    # Important values
    s=len(M)
    chi_coeff=3.0+0.01j
    chi=chi_coeff*np.identity(s)
    chi_invdag = np.linalg.inv(np.matrix.conjugate(np.transpose(chi)))
    ei_tr = np.matrix.conjugate(np.transpose(ei)) 
    Gdag=np.matrix.conjugate(np.transpose(M))
    print("Solve using the ",tsolver," method and the ",lsolver," method")
    start = timer()
    x_th=lsolverpick(lsolver,x0,tsolver,ei,ei_tr,chi_invdag,Gdag, Pv, e_vac)
    print("Done solving")
    mult=x_th[0]
    gradient=x_th[1]
    dual=x_th[2]
    objective=x_th[3]
    T = x_th[4]
    A = x_th[5]
    b = x_th[6]

    stop = timer()
    runtime=stop-start
    # print("x_cg",x_cg,"\n")
    print("Gradient value: ",gradient,"\n")
    print("Dual value: ",dual,"\n")
    print("Objective value: ",objective,"\n")
    print("The multipliers are: ", mult,"\n" )
    print("T: ", T,"\n" )
    print("A: ", A,"\n" )
    print("b: ", b,"\n" )
    # print("Sln",x_cg,"\n")
    print("Run time: ",runtime,"seconds \n")
    return runtime
# Here are some global values
Z=376.730313668 # Impedence of free space
wvlgth = 1.0
Qabs = 1e4
omega = (2*np.pi/wvlgth) * (1+1j/2/Qabs)
k0 = omega / 1
dL = 0.05
# 2 projection matrices, so 4 constraints
# cg and bfgs
dx=2 # -> Mx
dy=2 # -> My 
addx=1
liste=[[1,1,1,1,1,1,1,1]]
# liste=[[4,4,4,4,4,4,4,4]]
# liste=[[1,1,1,1,0,0,0,0],[0,0,0,0,1,1,1,1]]
# liste=[[1,1,1,1,1,1,1,1],[5,6,7,8,1,2,3,4]]
# liste=[[1,2,3,4,5,6,7,8],[5,6,7,8,1,2,3,4]]
setup = DipoleOverSlab_setup(dx,dy,addx,liste,dL, wvlgth)
print("setup",setup)
M=setup[0]
# print("M shape",M.shape,"\n")
ei = setup[1]
print("ei", ei,"\n")
Pv = setup[2]
e_vac=setup[3]
print("e_vac",e_vac,"\n")
x0 = np.zeros(2*len(liste),dtype='complex')
x0[0] = 0.1
# x0[1] = 1.1
tsolver = "th"
lsolver = "item"
bfgs_time = []
opt=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv, e_vac)


dx=2 # -> Mx
dy=2 # -> My 
addx=1
liste=[[1,1,1,1,1,1,1,1]]
setup=DipoleOverSlab_setup(dx,dy,addx,liste,dL, wvlgth)
print("setup",setup)
M=setup[0]
# print("M shape",M.shape,"\n")
ei=setup[1]
Pv=setup[2]
e_vac=setup[3]
x0_2=np.zeros(2*len(liste),dtype='complex')
x0_2[0]=0.15
x0_2[1]=0.05
tsolver="th"
lsolver="bfgs"
opt_2=extracted_power_optmization(M,ei,x0,tsolver,lsolver,Pv, e_vac)
# Finite difference calculation
diff = 0.05
#np.real(x0_2 - x0)
fin_diff = (opt-opt_2)/diff
print("fin_diff",fin_diff ,"\n")