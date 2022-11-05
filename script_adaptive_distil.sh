#! /bin/bash
#SBATCH --job-name adaptive-distil
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --partition ai4bp
#SBATCH --time 07-00:00:00
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-task 4
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090

srun fairseq-train indic-en-exp/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 16384 \
--arch transformer_1x_v0 \
--dropout 0.2 \
--task translation \
--kd-strategy word_and_seq_level \
--teacher-checkpoint-path checkpoints/indicTrans/checkpoint_best.pt \
--criterion label_smoothed_cross_entropy_with_kd \
--label-smoothing 0.1 \
--use-adaptive-weightage \
--adaptive-smoothing 0.5 \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--save-dir checkpoints/adaptive-distil \
--save-interval 1 \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 4 \
--num-workers 64 \
--memory-efficient-fp16 \
--wandb-project Indic-En-Distillation \
--user-dir indicTrans/model_configs > logs/adaptive_distil.log