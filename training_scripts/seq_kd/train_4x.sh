#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 16
#SBATCH --gpus-per-task 8
#SBATCH --partition ai4bp
#SBATCH --time=07-00:00:00
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090

srun fairseq-train ../../../data_bin/v2_distilled_indic_en_bin/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer_4x \
--activation-fn gelu \
--encoder-normalize-before \
--decoder-normalize-before \
--layernorm-embedding \
--dropout 0.2 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--save-dir ../../checkpoints/4x \
--save-interval 1 \
--save-interval-updates 5000 \
--keep-interval-updates 0 \
--no-epoch-checkpoints \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 8 \
--num-workers 16 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--wandb-project Indic-En-Distillation \
