#/bin/bash
echo `date`

job_name=$1
n_gpus=${2:-1}
n_cpus=${3:-16}

echo -e "[INFO]\tstarting ${job_name} in interactive mode and requesting ${n_gpus} gpus and ${n_cpus} cpus"
srun --nodes=1 \
     --job-name=$job_name \
     --ntasks-per-node=1 \
     --partition=ai4bp \
     --time=03-00:00:00 \
     --cpus-per-task=$n_cpus \
     --gpus-per-task=$n_gpus \
     --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090 \
     --pty \
     /bin/bash

echo -e "[INFO]\tresources allocated!"