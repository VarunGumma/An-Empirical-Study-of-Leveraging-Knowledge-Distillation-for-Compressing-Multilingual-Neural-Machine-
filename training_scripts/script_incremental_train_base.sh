#/bin/bash

fairseq-train $1/v2_0_binarized/final_bin:$1/v2_10_binarized/final_bin:$1/v2_20_binarized/final_bin:$1/v2_30_binarized/final_bin:$1/v2_40_binarized/final_bin:$1/v2_50_binarized/final_bin:$1/v2_60_binarized/final_bin:$1/v2_70_binarized/final_bin:$1/v2_80_binarized/final_bin:$1/v2_90_binarized/final_bin:$1/v2_100_binarized/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-epoch 11 \
--max-tokens 16384 \
--arch transformer_1x_v0 \
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
--save-dir ../checkpoints/base_with_best_bleu_incremental \
--save-interval 1 \
--keep-last-epochs 1 \
--patience 11 \
--skip-invalid-size-inputs-valid-test \
--validate-interval-updates 10000 \
--update-freq 1 \
--distributed-world-size 4 \
--user-dir ../model_configs \
--num-workers 64 \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--memory-efficient-fp16 \
--wandb-project Indic-En-Distillation
