#! /bin/bash
#SBATCH --job-name vanilla-distil
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition ai4bp
#SBATCH --time 07-00:00:00
#SBATCH --cpus-per-task 32
#SBATCH --gpus-per-task 4
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090

srun fairseq-train indic-en-exp/final_bin \
--max-source-positions 210 --max-target-positions 210 --max-update 1000000 --max-tokens 16384 \
--arch transformer --activation-fn gelu --layernorm-embedding --decoder-normalize-before --encoder-normalize-before --dropout 0.2 \
--task kd_translation --kd-strategy word_and_seq_level --teacher-checkpoint-path indic-en/model_4x_checkpoint_best.pt \
--criterion kd_label_smoothed_cross_entropy --label-smoothing 0.1 --alpha 0.5 --teacher-temp 2.5 --student-temp 2.5 \
--source-lang SRC --target-lang TGT \
--lr-scheduler inverse_sqrt --optimizer adam --adam-betas "(0.9, 0.98)" --clip-norm 1.0 --warmup-init-lr 1e-07 --lr 0.0005 --warmup-updates 4000 \
--save-dir indic-en-exp/model-1x-gelu-norm-distil --save-interval 1 --keep-last-epochs 1 --patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 4 \
--wandb-project Indic-En-Distillation \
--user-dir indicTrans/model_configs
