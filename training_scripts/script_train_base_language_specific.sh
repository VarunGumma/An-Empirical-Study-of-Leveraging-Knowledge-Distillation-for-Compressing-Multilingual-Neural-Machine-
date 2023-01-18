lang=$1
type=$2

fairseq-train ../../data_dir/bilingual/$type/$lang/final_bin \
--max-source-positions 210 \
--max-target-positions 210 \
--max-update 1000000 \
--max-tokens 4096 \
--arch transformer_1x_v0 \
--dropout 0.1 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--source-lang SRC \
--target-lang TGT \
--lr-scheduler inverse_sqrt \
--optimizer adam \
--adam-betas "(0.9, 0.98)" \
--clip-norm 1.0 \
--warmup-init-lr 1e-09 \
--lr 7e-4 \
--warmup-updates 120 \
--save-dir ../checkpoints/base_${lang}_${type} \
--save-interval 1 \
--keep-interval-updates 0 \
--no-epoch-checkpoints \
--patience 5 \
--skip-invalid-size-inputs-valid-test \
--update-freq 8 \
--distributed-world-size 1 \
--num-workers 32 \
--user-dir ../model_configs \
--eval-bleu \
--eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe \
--eval-bleu-print-samples \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--wandb-project Indic-En-Distillation \
--memory-efficient-fp16 
