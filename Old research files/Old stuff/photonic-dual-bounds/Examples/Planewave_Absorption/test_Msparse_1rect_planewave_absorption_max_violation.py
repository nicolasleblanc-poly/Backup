import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la

import time,sys,argparse

from get_planewave_absorption_Msparse_bounds import get_Msparse_1designrect_normal_absorption, get_Msparse_1designrect_normal_absorption_iterative_max_violation


parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store',type=float,default=1.0)
parser.add_argument('-ReChi',action='store',type=float,default=2.0)
parser.add_argument('-ImChi',action='store',type=float,default=1e-2)
parser.add_argument('-gpr',action='store',type=int,default=20)
parser.add_argument('-design_x',action='store',type=float,default=1.0)
parser.add_argument('-design_y',action='store',type=float,default=1.0)
parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-iter_period',action='store',type=int,default=20)

parser.add_argument('-name',action='store',type=str,default='test')


args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi

optLags, optgrad, dualval, objval, sigma_abs, sigma_enh = get_Msparse_1designrect_normal_absorption_iterative_max_violation(chi, args.wavelength, args.design_x, args.design_y, args.pml_sep, args.pml_thick, args.gpr, iter_period=args.iter_period, name=args.name)


print('optLags', optLags)
print('optgrad', optgrad)
print('dualval', dualval, 'objval', objval)
print('sigma_abs', sigma_abs)
print('sigma_enh', sigma_enh)

