#! /bin/bash
#SBATCH --job-name evaluate-benchmarks
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --gpus-per-task 1
#SBATCH --partition ai4bp
#SBATCH --time 00-06:00:00
#SBATCH --cpus-per-task 64

srun evaluate_benchmarks.sh \
     indicTrans \
     baseline \
     distil \
     adaptive-distil \
     batch-distil \
     global-distil \
     global-multi-distil \
     global-multi-adaptive-distil > ../logs/evaluate_benchmarks.log