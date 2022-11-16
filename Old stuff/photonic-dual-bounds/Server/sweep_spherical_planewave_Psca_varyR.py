import mpmath
from mpmath import mp
import time,sys,argparse
import numpy as np

sys.path.append('/u/pengning/Photonic_Dual_Bounds/photonic-dual-bounds/')

from spatialProjopt_multiRegion_spherical_planewave_Psca_numpy import sweep_Psca_spherical_planewave_spatialProjlists_varyR


parser = argparse.ArgumentParser()
parser.add_argument('-wavelength',action='store', type=float, default=1.0)
parser.add_argument('-pow10Rlow',action='store',type=float,default=-1.0)
parser.add_argument('-pow10Rhigh',action='store',type=float,default=1.0)
parser.add_argument('-numberofpoints',action='store',type=int,default=100)
parser.add_argument('-mpdps',action='store',type=int,default=50)
parser.add_argument('-ReChi',action='store',type=float,default=1.0)
parser.add_argument('-ImChi',action='store',type=float,default=1.0)
parser.add_argument('-Minitklim',action='store',type=int,default=25)
parser.add_argument('-Ninitklim',action='store',type=int,default=25)
parser.add_argument('-incPIm',action='store',type=int, default=-1)
parser.add_argument('-sqrnormtol',action='store',type=float, default=1e-8)
parser.add_argument('-Unormtol',action='store',type=float, default=1e-12)

args,unknown = parser.parse_known_args()
print('the parameters for this run:',flush=True)
for arg in vars(args):
    print(arg,'is',getattr(args,arg),flush=True)

mp.dps = args.mpdps
Rlist = 10.0**np.linspace(args.pow10Rlow,args.pow10Rhigh,args.numberofpoints) #from small to big so we see preliminary results early
filename = 'planewave_spatialProjlist_incPIm{:d}_Psca_dps{:d}_Chi{:0.2f}+{:0.2e}j_wvlgth{:0.2f}_R{:0.3f}-{:0.3f}'.format(args.incPIm, mp.dps, args.ReChi,args.ImChi, args.wavelength, Rlist[0], Rlist[-1])

tic = time.time() #check how long the runtime is
k = 2*np.pi/args.wavelength
incPIm = args.incPIm>0
chi = args.ReChi + 1j*args.ImChi

RPfaclist = [0.2, 0.4, 0.6, 0.8,1.0]

currentRlist, duallist, objlist = sweep_Psca_spherical_planewave_spatialProjlists_varyR(k, Rlist, chi, RPfaclist, incPIm=incPIm, Minitklim=args.Minitklim, Ninitklim=args.Ninitklim, normtol=args.sqrnormtol, Unormtol=args.Unormtol, filename=filename, feedforward=True)

duallist = np.array(duallist)
objlist = np.array(objlist)
currentRlist = np.array(currentRlist)

np.save(filename+'_Rlist.npy',currentRlist)
np.save(filename+'_duallist.npy',duallist)
np.save(filename+'_objlist.npy',objlist)
print(time.time()-tic)
