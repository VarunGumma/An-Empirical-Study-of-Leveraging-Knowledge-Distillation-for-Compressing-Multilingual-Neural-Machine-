#/bin/bash
echo `date`

n_gpus=${1:-1}
n_cpus=${2:-16}

echo -e "[INFO]\tstarting job in interactive mode and requesting ${n_gpus} gpus and ${n_cpus} cpus"

if [ "$n_gpus" -eq "0" ]; then
     srun --nodes=1 \
          --ntasks-per-node=1 \
          --cpus-per-task=$n_cpus \
          --pty \
          /bin/bash
else
     srun --nodes=1 \
          --ntasks-per-node=1 \
          --cpus-per-task=$n_cpus \
          --gpus-per-task=$n_gpus \
          --pty \
          /bin/bash
fi

echo -e "[INFO]\tresources deallocated"