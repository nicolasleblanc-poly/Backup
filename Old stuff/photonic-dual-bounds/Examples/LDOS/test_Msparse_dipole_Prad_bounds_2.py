import numpy as np

import sys,argparse
sys.path.append('/home/rk9166/photonic-dual-bounds-editable/')

from get_Msparse_dipole_power_bounds_Green_2 import get_TM_dipole_Prad_Msparse_bound

parser = argparse.ArgumentParser()

parser.add_argument('-wavelength', action='store', type=float, default=1.0)
parser.add_argument('-ReChi', action='store', type=float, default=2.0)
parser.add_argument('-ImChi', action='store', type=float, default=1e-2)

parser.add_argument('-gpr', action='store', type=int, default=40)
parser.add_argument('-vacuum_x', action='store', type=float, default=0.05)
parser.add_argument('-vacuum_y', action='store', type=float, default=0.05)
parser.add_argument('-emitter_x', action='store', type=float, default=0.025)
parser.add_argument('-emitter_y', action='store', type=float, default=0.025)
parser.add_argument('-design_x', action='store', type=float, default=1.0)
parser.add_argument('-design_y', action='store', type=float, default=1.0)
parser.add_argument('-dist', action='store', type=float, default=0.0)

parser.add_argument('-pml_sep',action='store',type=float,default=0.5)
parser.add_argument('-pml_thick',action='store',type=float,default=0.5)

parser.add_argument('-NProjx',action='store',type=int,default=1)
parser.add_argument('-NProjy',action='store',type=int,default=1)
# parser.add_argument('-NProjvx',action='store',type=int,default=1)
# parser.add_argument('-NProjvy',action='store',type=int,default=1)


args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

chi = args.ReChi + 1j*args.ImChi

# Prad_enh = get_TM_dipole_Prad_Msparse_bound(chi, args.wavelength, args.design_x, args.design_y, args.vacuum_x, args.vacuum_y, args.dist, args.pml_sep, args.pml_thick, args.gpr, args.NProjx, args.NProjy, args.NProjvx, args.NProjvy)
Prad_enh = get_TM_dipole_Prad_Msparse_bound(chi, args.wavelength, args.design_x, args.design_y, args.vacuum_x, args.vacuum_y, args.emitter_x, args.emitter_y, args.dist, args.pml_sep, args.pml_thick, args.gpr, args.NProjx, args.NProjy)
# Prad_enh = get_TM_dipole_Prad_Msparse_bound(chi, args.wavelength, args.design_x, args.design_y, args.dist, args.pml_sep, args.pml_thick, args.gpr, args.NProjx, args.NProjy)

print('bound for Prad enhancement', Prad_enh)
