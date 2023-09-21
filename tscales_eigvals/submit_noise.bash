#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 10 # number of cores 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 0-12:00 # time (D-HH:MM) 
#SBATCH --output=out/out_%a.out                                         
#SBATCH --error=out/err_%a.out

module load python/3.7.3

python3 -u compute_noise_floor.py -kw ${SLURM_ARRAY_TASK_ID}


