#/bin/bash
echo `date`

n_gpus=${1:-1}
n_cpus=${2:-16}

echo -e "[INFO]\tstarting job in interactive mode and requesting ${n_gpus} gpus and ${n_cpus} cpus"

if [ "$n_gpus" -eq "0" ]; then
     srun --nodes=1 \
          --ntasks-per-node=1 \
          --partition=ai4bp \
          --time=07-00:00:00 \
          --cpus-per-task=$n_cpus \
          --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090 \
          --pty \
          /bin/bash
else
     srun --nodes=1 \
          --ntasks-per-node=1 \
          --partition=ai4bp \
          --time=07-00:00:00 \
          --cpus-per-task=$n_cpus \
          --gpus-per-task=$n_gpus \
          --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090 \
          --pty \
          /bin/bash
fi

echo -e "[INFO]\tresources deallocated"