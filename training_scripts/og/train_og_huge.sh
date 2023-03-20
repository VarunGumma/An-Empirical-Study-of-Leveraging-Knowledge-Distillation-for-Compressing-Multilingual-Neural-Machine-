#!/bin/bash

data_dir=$1
ckpt_dir=$2
wandb_project=$3

fairseq-train $data_dir \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 16384 \
--arch transformer \
--activation-fn gelu \
--encoder-normalize-before \
--decoder-normalize-before \
--layernorm-embedding \
--encoder-embed-dim 1536 \
--decoder-embed-dim 1536 \
--encoder-ffn-embed-dim 4096 \
--decoder-ffn-embed-dim 4096 \
--encoder-attention-heads 16 \
--decoder-attention-heads 16 \
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
--save-dir $ckpt_dir/og_wo_any_nway_huge \
--save-interval 1 \
--save-interval-updates 5000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--patience 10 \
--skip-invalid-size-inputs-valid-test \
--update-freq 2 \
--distributed-world-size 2 \
--num-workers 16 \
--wandb-project $wandb_project \
--memory-efficient-fp16  \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric
