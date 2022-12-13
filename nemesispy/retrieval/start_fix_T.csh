#!/bin/tcsh -f
#SBATCH --mail-user=jingxuan.yang@hertford.ox.ac.uk
#SBATCH --mail-type=ALL
#SBATCH --job-name=2pt5DfixT
#SBATCH --output=zzzfixT.txt
#SBATCH -p shared
#SBATCH --mem-per-cpu=500
#SBATCH -N1
#SBATCH -n48

source /etc/profile.d/modules.csh
source ~/.tcshrc

date

mpirun -n 48 python3 2_pt_5_fix_T.py > stuff_fix.out

date