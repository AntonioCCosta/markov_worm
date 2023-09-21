#!/bin/bash
#SBATCH -p compute # partition 
#SBATCH -N 1 # number of nodes 
#SBATCH -n 10 # number of cores 
#SBATCH --mem 32G # memory pool for all cores 
#SBATCH -t 0-02:00 # time (D-HH:MM) 
#SBATCH -o out_%A_%a.out # STDOUT 
#SBATCH -e err_%A_%a.out # STDERR 

module load python/3.7.3

python3 compute_acfs.py -worm ${SLURM_ARRAY_TASK_ID}


