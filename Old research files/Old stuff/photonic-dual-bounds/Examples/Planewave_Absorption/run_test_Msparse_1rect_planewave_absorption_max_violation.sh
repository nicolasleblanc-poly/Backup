#!/bin/bash
#SBATCH --job-name=1rect
#SBATCH --output=Msparse_planewave_absorption_chi3+1e-1j_gpr40_Des0d5by0d5_max_violation.txt
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=200:00:00
#SBATCH --mem-per-cpu=12000
#SBATCH --partition=photon-planck
#SBATCH --error=error.err

source ~/anaconda3/etc/profile.d/conda.sh
conda activate darpa-nac

prog=test_Msparse_1rect_planewave_absorption_max_violation.py

wavelength=1
ReChi=3
ImChi=1e-1
gpr=40

design_x=0.5
design_y=0.5

pml_sep=0.5
pml_thick=0.5

iter_period=50
name='chi3_1e-1j_Des0d5x0d5'

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
#mpiexec -n $SLURM_NTASKS_PER_NODE python test_pixel_basis_idmap_bounds.py
python $prog -wavelength $wavelength -ReChi $ReChi -ImChi $ImChi -gpr $gpr -design_x $design_x -design_y $design_y -pml_sep $pml_sep -pml_thick $pml_thick -iter_period $iter_period -name $name

