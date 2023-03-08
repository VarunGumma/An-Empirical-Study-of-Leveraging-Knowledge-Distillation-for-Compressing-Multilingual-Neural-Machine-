fairseq-train $1 \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 8192 \
--arch transformer \
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
--save-dir ../../checkpoints/it \
--save-interval 1 \
--save-interval-updates 5000 \
--keep-interval-updates 1 \
--no-epoch-checkpoints \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 1 \
--distributed-world-size 8 \
--num-workers 16 \
--wandb-project Indic-En-Distillation \
--memory-efficient-fp16  \
