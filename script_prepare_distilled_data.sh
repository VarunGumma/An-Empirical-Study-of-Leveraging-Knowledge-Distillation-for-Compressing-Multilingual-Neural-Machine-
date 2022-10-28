#! /bin/bash
#SBATCH --job-name prepare-distilled-data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition ai4bp
#SBATCH --time 00-06:00:00
#SBATCH --cpus-per-task 128

srun distil_sam.sh > logs/distil_sam.log
