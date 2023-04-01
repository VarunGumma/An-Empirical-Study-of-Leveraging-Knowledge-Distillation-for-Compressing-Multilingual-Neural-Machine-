#!/bin/bash

data_dir=$1
model_name=$2
wandb_project=$3

fairseq-train $data_dir/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--save-interval 1 \
--arch transformer_${model_name} \
--criterion label_smoothed_cross_entropy \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--label-smoothing 0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--dropout 0.2 \
--save-dir $data_dir/$model_name \
--keep-last-epochs 5 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--memory-efficient-fp16 \
--user-dir ../model_configs \
--wandb-project $wandb_project \
--update-freq 4 \
--distributed-world-size 1 \
--max-tokens 16384