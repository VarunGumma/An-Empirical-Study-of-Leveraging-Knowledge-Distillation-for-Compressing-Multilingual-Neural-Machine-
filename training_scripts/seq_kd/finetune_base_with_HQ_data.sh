#!/bin/bash

data_dir=$1
ckpt_dir=$2
wandb_project=${ckpt_dir#*-}

fairseq-train $data_dir \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--save-interval 1 \
--save-interval-updates 5000 \
--arch transformer \
--activation-fn gelu \
--encoder-normalize-before \
--decoder-normalize-before \
--layernorm-embedding \
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
--save-dir $ckpt_dir/HQ-base \
--no-epoch-checkpoints \
--keep-interval-updates 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 8 \
--max-tokens 3072 \
--lr 3e-5 \
--restore-file $ckpt_dir/base/checkpoint_best.pt \
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
