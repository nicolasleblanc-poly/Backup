#!/bin/bash
#SBATCH --job-name=TEdip
#SBATCH --output=Split_TE_dipole_Msparse_oneside_Prad_dist0d2_chi5+1e-2j_gprx25_gpry25_Des0d4by2d0.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --partition=photon-planck
#SBATCH --error=error.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate darpa-nac

prog=split_Msparse_TE_dipole_oneside_Prad_bounds.py

wavelength=1
ReChi=5
ImChi=1e-2

orient='x'

gprx=25
gpry=25

dist=0.2

design_x=0.4
design_y=2.0

pml_sep=0.5
pml_thick=0.5

alg='Newton'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#mpiexec -n $SLURM_NTASKS_PER_NODE python test_pixel_basis_idmap_bounds.py
python $prog -wavelength $wavelength -ReChi $ReChi -ImChi $ImChi -orient $orient -gprx $gprx -gpry $gpry -design_x $design_x -design_y $design_y -dist $dist -pml_sep $pml_sep -pml_thick $pml_thick -alg $alg

