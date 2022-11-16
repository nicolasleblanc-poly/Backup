#!/bin/bash
#SBATCH --job-name=R16I1m2
#SBATCH --output=planewave_dualopt_ReChi16_ImChi1m2_Psca_varyR.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=64000
#SBATCH --error=R16Im2_ff1varyR.err
#SBATCH --partition=photon-planck
##SBATCH --mail-type=begin
##SBATCH --mail-type=end
##SBATCH --mail-user=pengning@princeton.edu

source ~/anaconda3/etc/profile.d/conda.sh
conda activate svbounds

prog=sweep_spherical_planewave_Psca_varyR.py

wavelength=1.0
pow10Rlow=-2
pow10Rhigh=-1
numberofpoints=2
mpdps=70
ReChi=16
ImChi=1e-2
Minitklim=35
Ninitklim=35
incPIm=-1
sqrnormtol=1e-8
Unormtol=1e-12

python $prog -wavelength $wavelength -pow10Rlow $pow10Rlow -pow10Rhigh $pow10Rhigh -numberofpoints $numberofpoints -mpdps $mpdps -ReChi $ReChi -ImChi $ImChi -Minitklim $Minitklim -Ninitklim $Ninitklim -incPIm $incPIm -sqrnormtol $sqrnormtol -Unormtol $Unormtol
