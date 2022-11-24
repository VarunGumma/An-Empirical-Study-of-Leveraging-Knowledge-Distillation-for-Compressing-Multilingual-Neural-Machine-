#/bin/bash

fairseq-train $1/v2_0_binarized \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer_1x_v0 \
--dropout 0.2 \
--task translation \
--kd-strategy global_multi_level \
--teacher-checkpoint-path ../checkpoints/indicTrans/checkpoint_best.pt \
--criterion label_smoothed_cross_entropy_with_kd \
--label-smoothing 0.1 \
--alpha 1 \
--use-adaptive-kd-rates \
--kd-queue-size 50000 \
--kd-selection-temp 1.5 \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-07 \
--lr 0.0005 \
--warmup-updates 4000 \
--save-dir ../checkpoints/base-global-multi-adaptive-distil \
--save-interval 1 \
--keep-last-epochs 5 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--run-sanity-val-steps \
--update-freq 1 \
--distributed-world-size 8 \
--num-workers 16 \
--wandb-project Indic-En-Distillation \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--user-dir ../model_configs