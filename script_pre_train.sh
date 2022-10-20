#! /bin/bash
#SBATCH --job-name distil-sam
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition ai4bp
#SBATCH --time 00-06:00:00
#SBATCH --cpus-per-task 128

srun pre_train.sh