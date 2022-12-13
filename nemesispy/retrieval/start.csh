#!/bin/tcsh -f
#SBATCH --mail-user=jingxuan.yang@hertford.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=2pt5D
#SBATCH --output=zzz.txt
#SBATCH -p shared
#SBATCH --mem-per-cpu=500
#SBATCH -N1
#SBATCH -n64

source /etc/profile.d/modules.csh
source ~/.tcshrc

date

mpirun -n 64 python3 2_pt_5_c.py > stuff.out

date