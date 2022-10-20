#! /bin/bash
#SBATCH --job-name eval
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 2
#SBATCH --partition ai4bp
#SBATCH --time 00-12:00:00
#SBATCH --cpus-per-task 32

srun eval.sh \
     1x-gelu-norm \
     1x-gelu-norm-distil \
     1x-gelu-norm-adaptive-distil \
     1x-gelu-norm-batch-distil \
     1x-gelu-norm-global-distil \
     1x-gelu-norm-global-multi-distil \
     1x-gelu-norm-global-multi-adaptive-disti:l