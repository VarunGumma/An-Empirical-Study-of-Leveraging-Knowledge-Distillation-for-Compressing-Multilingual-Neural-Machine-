#!/bin/bash

data_dir=$1
pretrained_ckpt=$2
model_name=$3
wandb_project=$4

fairseq-train $data_dir/final_bin \
    --max-source-positions 210 \
    --max-target-positions 210 \
    --max-update 1000000 \
    --save-interval 1 \
    --save-interval-updates 5000 \
    --arch transformer_$model_name \
    --criterion label_smoothed_cross_entropy \
    --source-lang SRC \
    --target-lang TGT \
    --label-smoothing 0.1 \
    --lr-scheduler inverse_sqrt \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.2 \
    --save-dir $data_dir/${model_name}_hq_ft \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \
    --patience 5 \
    --skip-invalid-size-inputs-valid-test \
    --update-freq 1 \
    --distributed-world-size 4 \
    --max-tokens 6144 \
    --lr 3e-5 \
    --restore-file $pretrained_ckpt/checkpoint_best.pt \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-dataloader \
    --reset-optimizer \
    --num-workers 16 \
    --wandb-project $wandb_project \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --eval-bleu-print-samples \
    --best-checkpoint-metric bleu \
    --memory-efficient-fp16 \
    --maximize-best-checkpoint-metric
