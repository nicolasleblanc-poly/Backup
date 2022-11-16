import numpy as np
import autograd.numpy as npa

import copy

import matplotlib as mpl
mpl.rcParams['figure.dpi']=100

import matplotlib.pylab as plt

from autograd.scipy.signal import convolve as conv
from skimage.draw import circle

import ceviche
from ceviche import fdfd_ez, jacobian
from ceviche.optimizers import adam_optimize
from ceviche.modes import insert_mode

import collections

# Define simulation parameters (see above)
omega = 2*np.pi*3e8/2.5e-6
dl = 25e-9

Nx = 160
Ny = 160
Npml = 20

epsr_min = 1.0
epsr_max = 4.0
blur_radius = 5
N_blur = 1
eta=0.5
N_proj=1
# Number of epochs in the optimization 
Nsteps=1000
# Step size for the Adam optimizer
step_size=1e-2
    

# Define permittivity for a straight waveguide
epsr = epsr_max*np.ones((Nx, Ny))  
epsr0 = np.ones((Nx, Ny))  
epsr[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 1.0

# Source position 
src_y = [Ny//2]
src_x = [Nx//2]
source = np.zeros((Nx, Ny), dtype=np.complex128)
source[src_x,src_y] = 1


# Run the simulation exciting mode 1
simulation = ceviche.fdfd_ez(omega, dl, epsr, [Npml, Npml])
Hx, Hy, Ez = simulation.solve(source)
simulation0 = ceviche.fdfd_ez(omega, dl, epsr0, [Npml, Npml])
Hx0, Hy0, Ez0 = simulation0.solve(source)
epsr_init = epsr_max
space=10
E0 = npa.real(Ez0[Nx//2,Ny//2])

# Projection that drives rho towards a "binarized" design with values either 0 or 1 
def operator_proj(rho, eta=0.5, beta=100, N=1):
    """Density projection
    eta     : Center of the projection between 0 and 1
    beta    : Strength of the projection
    N       : Number of times to apply the projection
    """
    for i in range(N):
        rho =  npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)), 
                          npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))
    
    return rho

# Blurring filter that results in smooth features of the structure
# First we define a function to create the kernel
def _create_blur_kernel(radius):
    """Helper function used below for creating the conv kernel"""
    rr, cc = circle(radius, radius, radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float64)
    kernel[rr, cc] = 1
    return  kernel/kernel.sum()

# Then we define the function to apply the operation
def operator_blur(rho, radius=2, N=1):
    """Blur operator implemented via two-dimensional convolution
    radius    : Radius of the circle for the conv kernel filter
    N         : Number of times to apply the filter
    
    Note that depending on the radius, the kernel is not always a
    perfect circle due to "pixelation" / stair casing
    """
    
    kernel = _create_blur_kernel(radius)
    
    for i in range(N):
        # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
        rho = conv(rho, kernel, mode='full')[radius:-radius,radius:-radius]
    
    return rho

def callback_output_structure(iteration, of_list, rho):
    """Callback function to output fields and the structures (for making sweet gifs)"""
    rho = rho.reshape((Nx, Ny))
    epsr = epsr_parametrization(rho, bg_rho, design_region, \
                                  radius=blur_radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
    simulation = viz_sim(epsr, source,'tmp3/epsr_%03d.png' % iteration)
    
def callback_beta_schedule(iteration, of_list, rho):
    """Callback function for the optimizer to schedule changes to beta with the iteration number"""
    
    # I am commiting a terrible sin by using globals here, but I am feeling lazy...
    global beta
    
    if iteration < 1000:
        beta = 0.0
    
    # Chain with the output structure callback
    callback_output_structure(iteration, of_list, rho)
def init_domain(Nx, Ny, Npml, space=10):
    rho = np.zeros((Nx, Ny))
    bg_rho = np.zeros((Nx, Ny))
    design_region = np.zeros((Nx, Ny))
    
    #bg_epsr = np.zeros((Nx, Ny))
    #epsr = np.zeros((Nx, Ny))
    #design_region = np.zeros((Nx, Ny))
    design_region[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 1
    bg_rho[Npml+space:Nx-Npml-space,Npml+space:Ny-Npml-space] = 1
    rho[Npml+space:Nx-Npml-space,Npml+space:Ny-Npml-space] = 0.5

    design_region[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    bg_rho[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    rho[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    return rho, bg_rho, design_region

designMask = np.zeros((Nx,Ny), dtype=np.bool)
designMask[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = True
designMask[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = False

def mask_combine_rho(rho, bg_rho, design_region):
    return rho*design_region + bg_rho*(design_region==0).astype(np.float64)
def viz_sim(epsr,source,fname):
    simulation = fdfd_ez(omega, dl, epsr, [Npml, Npml])
    Hx, Hy, Ez = simulation.solve(source)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6,3))
    #ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
    #for sl in slices:
    #    ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
    #ceviche.viz.abs(epsr, ax=ax, cmap='Greys');
    i = ax.imshow(epsr, cmap='Greys', interpolation='nearest')
    plt.savefig(fname,dpi=70)
    plt.close()
    return simulation
def epsr_parametrization(rho, bg_rho, design_region, radius=2, N_blur=1, beta=100, eta=0.5, N_proj=1):
    """Defines the parameterization steps for constructing rho
    """
    # Combine rho and bg_rho; Note: this is so the subsequent blur sees the waveguides
    rho = mask_combine_rho(rho, bg_rho, design_region)
    
    #rho = operator_blur(rho, radius=radius, N=N_blur)
    #rho = operator_proj(rho, beta=beta, eta=eta, N=N_proj)
    
    # Final masking undoes the blurring of the waveguides
    #rho = mask_combine_rho(rho, bg_rho, design_region)
    
    return epsr_min + (epsr_max-epsr_min) * rho
def objective(rho):
    rho = rho.reshape((Nx,Ny))
    epsr = epsr_parametrization(rho, bg_rho, design_region, \
                             radius=blur_radius, N_blur=N_blur, eta=eta, N_proj=N_proj)
    simulation.eps_r = epsr
    _, _, Ez = simulation.solve(source)
    return npa.real(Ez[Nx//2,Ny//2])/E0
    
rho, bg_rho, design_region = init_domain(Nx, Ny, Npml, space=space)

objective_jac = jacobian(objective, mode='reverse')
(rho_optimum, loss) = adam_optimize(objective, rho.flatten(), objective_jac, Nsteps=Nsteps, direction='max',bounds=[0.0,1.0], step_size=step_size)

rho_optimum = rho_optimum.reshape((Nx,Ny))
epsr = epsr_parametrization(rho_optimum, bg_rho, design_region, \
                            radius=blur_radius, N_blur=N_blur, eta=eta, N_proj=N_proj)
print(objective(rho_optimum),flush=True)
print(np.max(np.max(epsr)),np.min(np.min(epsr)),flush=True)

simm = viz_sim(epsr,source,"ReChi3_grey_omega.pdf")

# Define simulation parameters (see above)
omega = 2*np.pi*3e8/2.5e-6
dl = 25e-9

Nx = 160
Ny = 160
Npml = 20

epsr_min = 1.0
epsr_max = 4.0
blur_radius = 5
N_blur = 1
beta=500.0
eta=0.5
N_proj=1
# Number of epochs in the optimization 
Nsteps=1000
# Step size for the Adam optimizer
step_size=1e-2

# Define permittivity for a straight waveguide
epsr = epsr_max*np.ones((Nx, Ny))  
epsr0 = np.ones((Nx, Ny))  
epsr[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 1.0

# Source position 
src_y = [Ny//2]
src_x = [Nx//2]
source = np.zeros((Nx, Ny), dtype=np.complex128)
source[src_x,src_y] = 1


# Run the simulation exciting mode 1
simulation = ceviche.fdfd_ez(omega, dl, epsr, [Npml, Npml])
Hx, Hy, Ez = simulation.solve(source)
simulation0 = ceviche.fdfd_ez(omega, dl, epsr0, [Npml, Npml])
Hx0, Hy0, Ez0 = simulation0.solve(source)
epsr_init = epsr_max
space=10
E0 = npa.real(Ez0[Nx//2,Ny//2])

# Projection that drives rho towards a "binarized" design with values either 0 or 1 
def operator_proj(rho, eta=0.5, beta=100, N=1):
    """Density projection
    eta     : Center of the projection between 0 and 1
    beta    : Strength of the projection
    N       : Number of times to apply the projection
    """
    for i in range(N):
        rho =  npa.divide(npa.tanh(beta * eta) + npa.tanh(beta * (rho - eta)), 
                          npa.tanh(beta * eta) + npa.tanh(beta * (1 - eta)))
    
    return rho

# Blurring filter that results in smooth features of the structure
# First we define a function to create the kernel
def _create_blur_kernel(radius):
    """Helper function used below for creating the conv kernel"""
    rr, cc = circle(radius, radius, radius+1)
    kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float64)
    kernel[rr, cc] = 1
    return  kernel/kernel.sum()

# Then we define the function to apply the operation
def operator_blur(rho, radius=2, N=1):
    """Blur operator implemented via two-dimensional convolution
    radius    : Radius of the circle for the conv kernel filter
    N         : Number of times to apply the filter
    
    Note that depending on the radius, the kernel is not always a
    perfect circle due to "pixelation" / stair casing
    """
    
    kernel = _create_blur_kernel(radius)
    
    for i in range(N):
        # For whatever reason HIPS autograd doesn't support 'same' mode, so we need to manually crop the output
        rho = conv(rho, kernel, mode='full')[radius:-radius,radius:-radius]
    
    return rho

def callback_output_structure(iteration, of_list, rho):
    """Callback function to output fields and the structures (for making sweet gifs)"""
    rho = rho.reshape((Nx, Ny))
    epsr = epsr_parametrization(rho, bg_rho, design_region, \
                                  radius=blur_radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
    simulation = viz_sim(epsr, source,'tmp3/epsr_%03d.png' % iteration)
    
def callback_beta_schedule(iteration, of_list, rho):
    """Callback function for the optimizer to schedule changes to beta with the iteration number"""
    
    # I am commiting a terrible sin by using globals here, but I am feeling lazy...
    global beta
    
    if iteration < 500:
        beta = 10
    elif 500 <= iteration & iteration < 750:
        beta = 100
    elif 750 <= iteration & iteration < 1000:
        beta = 200
    else:
        beta = 300
    
    # Chain with the output structure callback
    callback_output_structure(iteration, of_list, rho)
def init_domain(Nx, Ny, Npml, space=10):
    rho = np.zeros((Nx, Ny))
    bg_rho = np.zeros((Nx, Ny))
    design_region = np.zeros((Nx, Ny))
    
    #bg_epsr = np.zeros((Nx, Ny))
    #epsr = np.zeros((Nx, Ny))
    #design_region = np.zeros((Nx, Ny))
    design_region[Npml+space:Nx-Npml-space, Npml+space:Ny-Npml-space] = 1
    bg_rho[Npml+space:Nx-Npml-space,Npml+space:Ny-Npml-space] = 1
    rho[Npml+space:Nx-Npml-space,Npml+space:Ny-Npml-space] = 0.5

    design_region[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    bg_rho[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    rho[Nx//2-18:Nx//2+18,Ny//2-18:Ny//2+18] = 0
    return rho, bg_rho, design_region

def mask_combine_rho(rho, bg_rho, design_region):
    return rho*design_region + bg_rho*(design_region==0).astype(np.float64)
def viz_sim(epsr,source,fname):
    simulation = fdfd_ez(omega, dl, epsr, [Npml, Npml])
    Hx, Hy, Ez = simulation.solve(source)
    fig, ax = plt.subplots(1, 1, constrained_layout=True, figsize=(6,3))
    #ceviche.viz.real(Ez, outline=epsr, ax=ax[0], cbar=False)
    #for sl in slices:
    #    ax[0].plot(sl.x*np.ones(len(sl.y)), sl.y, 'b-')
    #ceviche.viz.abs(epsr, ax=ax, cmap='Greys');
    i = ax.imshow(epsr, cmap='Greys', interpolation='nearest')
    plt.savefig(fname,dpi=70)
    plt.close()
    return simulation
def epsr_parametrization(rho, bg_rho, design_region, radius=2, N_blur=1, beta=100, eta=0.5, N_proj=1):
    """Defines the parameterization steps for constructing rho
    """
    # Combine rho and bg_rho; Note: this is so the subsequent blur sees the waveguides
    rho = mask_combine_rho(rho, bg_rho, design_region)
    
    #rho = operator_blur(rho, radius=radius, N=N_blur)
    rho = operator_proj(rho, beta=beta, eta=eta, N=N_proj)
    
    # Final masking undoes the blurring of the waveguides
    rho = mask_combine_rho(rho, bg_rho, design_region)
    
    return epsr_min + (epsr_max-epsr_min) * rho
def objective(rho):
    rho = rho.reshape((Nx,Ny))
    epsr = epsr_parametrization(rho, bg_rho, design_region, \
                             radius=blur_radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
    simulation.eps_r = epsr
    _, _, Ez = simulation.solve(source)
    return npa.real(Ez[Nx//2,Ny//2])/E0
    
rho, bg_rho, design_region = init_domain(Nx, Ny, Npml, space=space)

objective_jac = jacobian(objective, mode='reverse')
(rho_optimum, loss) = adam_optimize(objective, rho.flatten(), objective_jac, Nsteps=Nsteps, direction='max', step_size=step_size,callback=callback_beta_schedule)

rho_optimum = rho_optimum.reshape((Nx,Ny))
epsr = epsr_parametrization(rho_optimum, bg_rho, design_region, \
                            radius=blur_radius, N_blur=N_blur, beta=beta, eta=eta, N_proj=N_proj)
print(objective(rho_optimum),flush=True)
print(np.max(np.max(epsr)),np.min(np.min(epsr)),flush=True)
simm = viz_sim(epsr,source,"ReChi3_bin_omega.pdf")
