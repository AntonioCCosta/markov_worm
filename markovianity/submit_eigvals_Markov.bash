#!/bin/bash
#SBATCH -p compute
#SBATCH -t 08:00:00
#SBATCH -n 1
#SBATCH -c 16
#SBATCH --mem=16G                                    
#SBATCH --output=out/out_%a.out
#SBATCH --error=out/err_%a.out

module load python/3.7.3

python3 -u coarse_eigvals_Markov.py -kw ${SLURM_ARRAY_TASK_ID}
