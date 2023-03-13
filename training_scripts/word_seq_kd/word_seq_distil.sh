#!/bin/bash

data_dir=$1
ckpt_dir=$2
wandb_project=${ckpt_dir#*-}

fairseq-train $data_dir \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer \
--activation-fn gelu \
--encoder-normalize-before \
--decoder-normalize-before \
--layernorm-embedding \
--dropout 0.2 \
--task translation_with_kd \
--kd-strategy word_seq_level \
--teacher-checkpoint-path $ckpt_dir/it/checkpoint_best.pt \
--criterion label_smoothed_cross_entropy_with_kd \
--label-smoothing 0.1 \
--alpha 0.5 \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--save-dir $ckpt_dir/word_seq_distil \
--save-interval 1 \
--save-interval-updates 5000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 8 \
--num-workers 64 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--memory-efficient-fp16 \
--wandb-project $wandb_project