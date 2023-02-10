fairseq-train  ../../../data_bin/v2_distilled_indic_en_HQ_bin/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--save-interval 1 \
--save-interval-updates 5000 \
--arch transformer_4x \
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
--save-dir ../../checkpoints/HQ-4x \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 4 \
--max-tokens 6144 \
--lr 3e-5 \
--restore-file ../../checkpoints/4x/checkpoint_best.pt \
--reset-lr-scheduler \
--reset-meters \
--reset-dataloader \
--reset-optimizer \
--num-workers 10 \
--wandb-project Indic-En-Distillation \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--memory-efficient-fp16 \
--maximize-best-checkpoint-metric
