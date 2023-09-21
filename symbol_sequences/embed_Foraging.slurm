#!/bin/bash
#SBATCH -p short
#SBATCH -t 05:00:00
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --mem=4G                                    
#SBATCH --output=out_%A.out
#SBATCH --error=err_%A.out


module load python/3.7.3

python3 get_phspace_labels.py
