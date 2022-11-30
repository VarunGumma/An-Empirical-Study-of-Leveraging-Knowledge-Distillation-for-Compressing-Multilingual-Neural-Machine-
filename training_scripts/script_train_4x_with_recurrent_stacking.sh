#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-task 4
#SBATCH --partition ai4bp
#SBATCH --time=07-00:00:00
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090

fairseq-train ../../data_bin_dir/v2_0_binarized/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 16384 \
--arch transformer_4x_v0 \
--encoder-recurrent-stacking 6 \
--decoder-recurrent-stacking 6 \
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
--save-dir ../checkpoints/4x_RS_layers_with_best_bleu \
--save-interval 1 \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 4 \
--num-workers 64 \
--user-dir ../model_configs \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--wandb-project Indic-En-Distillation