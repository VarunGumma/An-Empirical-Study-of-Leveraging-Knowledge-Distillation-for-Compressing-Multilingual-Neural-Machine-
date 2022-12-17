#!/bin/bash

save_to_dir="V_base_with_adapters_finetuned_on"
restore_from_dir="base"

for lang in as bn gu hi kn ml mr or pa ta te; do
    echo `date`
    echo -e "\n[INFO]\tfinetuning on ${lang}"

    save_to_dir=${save_to_dir}_${lang}

    echo "restoring from ${restore_from_dir}"
    echo "saving checkpoints to ${save_to_dir}"

    fairseq-train ../../data_dir/v2_distilled_indic_en_language_wise_bin/$lang/final_bin \
    --max-source-positions 210 \
    --max-target-positions 210 \
    --max-update 1000000 \
    --save-interval 1 \
    --arch transformer_1x_v0 \
    --encoder-add-adapters \
    --encoder-adapter-reduction-factor-trend "[2, 4, 8, 8, 4, 2]" \
    --encoder-adapter-lang-ids "[\"as\", \"bn\", \"gu\", \"hi\", \"kn\", \"ml\", \"mr\", \"or\", \"pa\", \"ta\", \"te\"]" \
    --encoder-finetune-adapter $lang \
    --decoder-add-adapters \
    --decoder-adapter-reduction-factor-trend "[2, 4, 8, 8, 4, 2]" \
    --decoder-adapter-lang-ids "[\"as\", \"bn\", \"gu\", \"hi\", \"kn\", \"ml\", \"mr\", \"or\", \"pa\", \"ta\", \"te\"]" \
    --decoder-finetune-adapter $lang \
    --criterion label_smoothed_cross_entropy \
    --source-lang SRC \
    --lr-scheduler inverse_sqrt \
    --target-lang TGT \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 400 \
    --dropout 0.2 \
    --save-dir ../checkpoints/$save_to_dir \
    --keep-last-epochs 1 \
    --patience 5 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir ../model_configs \
    --update-freq 1 \
    --distributed-world-size 6 \
    --max-tokens 4096 \
    --lr 7.5e-4 \
    --restore-file ../checkpoints/$restore_from_dir/checkpoint_best.pt \
    --load-checkpoint-liberally \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-dataloader \
    --reset-optimizer \
    --num-workers 16 \
    --validate-interval-updates 500 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    --wandb-project Indic-En-Distillation

    restore_from_dir=$save_to_dir
done