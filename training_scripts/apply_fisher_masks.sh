#!/bin/bash

fairseq-train ../../data_bin/v2_distilled_indic_en_bin/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--arch transformer_1x_v0 \
--task translation_with_fisher_masks \
--criterion label_smoothed_cross_entropy \
--source-lang SRC \
--target-lang TGT \
--label-smoothing 0.1 \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--warmup-updates 4000 \
--lr-scheduler inverse_sqrt \
--dropout 0.2 \
--save-interval 1 \
--save-interval-updates 5000 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 8 \
--distributed-world-size 1 \
<<<<<<< HEAD
--max-tokens 8192 \
--lr 5e-4 \
=======
--max-tokens 2048 \
--lr 3e-5 \
>>>>>>> 8601bdf (.)
--restore-file ../checkpoints/base/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer \
--num-workers 32 \
--memory-efficient-fp16 \
--load-fisher-masks-from ../checkpoints/masks.pt