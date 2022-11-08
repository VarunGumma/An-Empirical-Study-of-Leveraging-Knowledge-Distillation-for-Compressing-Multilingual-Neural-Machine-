#/bin/bash
echo `date`

n_gpus=$1
n_cpus=${2:-64}

echo -e "[INFO]\tstarting interactive session and requesting ${n_gpus} gpus and ${n_cpus} cpus"
srun --nodes=1 \
     --ntasks-per-node=1 \
     --partition=ai4bp \
     --time=07-00:00:00 \
     --cpus-per-task=$n_cpus \
     --gpus-per-task=$n_gpus \
     --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090 \
     --pty \
     /bin/bash