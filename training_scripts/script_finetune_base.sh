#!/bin/bash

#SBATCH --nodes 1
#SBATCH --ntasks-per-node 1
#SBATCH --cpus-per-task 64
#SBATCH --gpus-per-task 4
#SBATCH --partition ai4bp
#SBATCH --time=07-00:00:00
#SBATCH --export=ALL,http_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090,https_proxy=http://dgx-proxy-mn.mgmt.siddhi.param:9090

srun fairseq-train ../../data_bin_dir/v2_0_HQ_binarized/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--save-interval 1 \
--arch transformer_1x_v0 \
--criterion label_smoothed_cross_entropy \
--source-lang SRC \
--lr-scheduler inverse_sqrt \
--target-lang TGT \
--label-smoothing 0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 \
--dropout 0.2 \
--save-dir ../checkpoints/baseline_with_best_bleu_finetuned_on_distilled_data_V2 \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--validate-interval-updates 10000 \
--user-dir ../model_configs \
--update-freq 1 \
--distributed-world-size 4 \
--max-tokens 8192 \
--lr 3e-5 \
--restore-file ../checkpoints/base_with_best_bleu/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer \
--num-workers 64 \
--wandb-project Indic-En-Distillation \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--memory-efficient-fp16
