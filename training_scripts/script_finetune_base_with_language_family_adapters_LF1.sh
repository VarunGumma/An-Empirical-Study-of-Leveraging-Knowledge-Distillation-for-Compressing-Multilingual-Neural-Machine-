#!/bin/bash

save_to_dir="base_with_language_family_adapters_LF1_finetuned_on"
restore_from_dir="base"

for lang in as-bn-or ta-te-ml-kn hi-pa-gu-mr; do
    echo `date`
    echo -e "\n[INFO]\tfinetuning on ${lang}"

    save_to_dir=${save_to_dir}_${lang}

    echo "restoring from ${restore_from_dir}"
    echo "saving checkpoints to ${save_to_dir}"

    fairseq-train ../../data_dir/v2_distilled_indic_en_language_family_LF1_bin/$lang/final_bin \
    --max-source-positions 210 \
    --max-target-positions 210 \
    --max-update 1000000 \
    --save-interval 1 \
    --arch transformer_1x_v0 \
    --encoder-add-adapters \
    --encoder-adapter-bottleneck-dim 256 \
    --encoder-adapter-langs as-bn-or,hi-pa-gu-mr,ta-te-ml-kn \
    --encoder-finetune-adapter $lang \
    --decoder-add-adapters \
    --decoder-adapter-bottleneck-dim 256 \
    --decoder-adapter-langs as-bn-or,hi-pa-gu-mr,ta-te-ml-kn \
    --decoder-finetune-adapter $lang \
    --adapter-activation-fn swish \
    --adapter-dropout 0.1 \
    --target-lang TGT \
    --criterion label_smoothed_cross_entropy \
    --source-lang SRC \
    --lr-scheduler inverse_sqrt \
    --label-smoothing 0.1 \
    --optimizer adam \
    --adam-betas "(0.9, 0.98)" \
    --clip-norm 1.0 \
    --warmup-init-lr 1e-07 \
    --warmup-updates 4000 \
    --dropout 0.2 \
    --save-dir ../checkpoints/$save_to_dir \
    --save-interval-updates 1000 \
    --keep-last-epochs 1 \
    --patience 5 \
    --skip-invalid-size-inputs-valid-test \
    --user-dir ../model_configs \
    --update-freq 4 \
    --distributed-world-size 1 \
    --max-tokens 16384 \
    --lr 5e-4 \
    --restore-file ../checkpoints/$restore_from_dir/checkpoint_best.pt \
    --load-checkpoint-liberally \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-dataloader \
    --reset-optimizer \
    --num-workers 6 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    
    restore_from_dir=$save_to_dir
done
