fairseq-train ../../data_bin/v2_hq_indic_en_bin/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer_4x \
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
--warmup-updates 2000 \
--save-dir ../checkpoints/it_hq \
--save-interval 1 \
--save-interval-updates 1000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 8 \
--distributed-world-size 1 \
--num-workers 32 \
--wandb-project Indic-En-Distillation \
--memory-efficient-fp16