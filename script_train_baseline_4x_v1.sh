fairseq-train indic-en-exp/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 16384 \
--arch transformer_4x_v0 \
--encoder-layers 1 \
--decoder-layers 1 \
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
--save-dir checkpoints/baseline-4x-v1 \
--save-interval 1 \
--keep-last-epochs 1 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--memory-efficient-fp16 \
--update-freq 4 \
--distributed-world-size 1 \
--num-workers 16 \
--user-dir indicTrans/model_configs \
--wandb-project Indic-En-Distillation > logs/baseline_4x_v1.log
