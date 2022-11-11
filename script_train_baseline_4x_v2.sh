fairseq-train indic-en-exp/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer_4x_v0 \
--encoder-layers 2 \
--decoder-layers 2 \
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
--save-dir checkpoints/baseline-4x-v2 \
--save-interval 1 \
--keep-last-epochs 5 \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 8 \
--num-workers 16 \
--user-dir indicTrans/model_configs \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \    
--wandb-project Indic-En-Distillation
