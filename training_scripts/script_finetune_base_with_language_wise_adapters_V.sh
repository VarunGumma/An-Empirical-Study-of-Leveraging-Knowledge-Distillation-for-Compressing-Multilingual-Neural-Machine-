#!/bin/bash

save_to_dir="V_base_with_language_wise_adapters_finetuned_on"
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
    --save-interval-updates 500 \
    --arch transformer \
    --activation-fn gelu \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --layernorm-embedding \
    --encoder-add-adapters \
    --encoder-adapter-bottleneck-dim-trend 256,128,64,64,128,256 \
    --encoder-adapter-langs as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
    --encoder-finetune-adapter $lang \
    --decoder-add-adapters \
    --decoder-adapter-bottleneck-dim-trend 256,128,64,64,128,256 \
    --decoder-adapter-langs as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
    --decoder-finetune-adapter $lang \
    --adapter-activation-fn swish \
    --adapter-dropout 0.1 \
    --criterion label_smoothed_cross_entropy \
    --source-lang SRC \
    --target-lang TGT \
    --lr-scheduler inverse_sqrt \
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
    --num-workers 32 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu

    restore_from_dir=$save_to_dir
done
