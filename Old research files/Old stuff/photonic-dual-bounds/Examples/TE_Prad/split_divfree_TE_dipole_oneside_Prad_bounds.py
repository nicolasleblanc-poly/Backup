import numpy as np

import time,sys,argparse

from get_TE_dipole_Prad_bounds import get_TE_dipole_oneside_Prad_divfree_iterative_splitting

parser = argparse.ArgumentParser()

parser.add_argument('-wavelength', action='store', type=float, default=1.0)
parser.add_argument('-ReChi', action='store', type=float, default=2.0)
parser.add_argument('-ImChi', action='store', type=float, default=1e-2)

parser.add_argument('-orient', action='store', type=str, default='x')

parser.add_argument('-divfree', action='store', type=int, default=1)

parser.add_argument('-gprx', action='store', type=int, default=20)
parser.add_argument('-gpry', action='store', type=int, default=20)

parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-dist', action='store', type=float, default=0.5)

parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-alg',action='store',type=str,default='Newton')

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi

divfree = args.divfree>0

Prad_enh = get_TE_dipole_oneside_Prad_divfree_iterative_splitting(chi, args.wavelength, args.orient, args.design_x, args.design_y, args.dist, args.pml_sep, args.pml_thick, args.gprx, args.gpry, divfree=divfree, alg=args.alg)

