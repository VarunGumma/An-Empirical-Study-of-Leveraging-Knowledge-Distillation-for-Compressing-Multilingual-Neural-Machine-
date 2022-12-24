#!/bin/bash

save_to_dir="VV_base_with_adapters_finetuned_on"
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
    --arch transformer_1x_v0 \
    --hyperadapter-langs as,bn,gu,hi,kn,ml,mr,or,pa,ta,te,en \
    --hyperadapter-src-lang $lang \
    --hyperadapter-tgt-lang en \
    --hyperadapter-dropout 0.1 \
    --hyperadapter-activation-fn relu \
    --encoder-add-hyperadapters \
    --encoder-hyperadapter-lang-embedding-dim 64 \
    --encoder-hyperadapter-layer-embedding-dim 64 \
    --encoder-hyperadapter-bottleneck-dim 64 \
    --encoder-hyperadapter-hidden-dim 64 \
    --encoder-hyperadapter-num-hidden-layers 2 \
    --encoder-hyperadapter-generate-layernorm \
    --encoder-hyperadapter-language-embedding-tied \
    --encoder-hyperadapter-inputs src,tgt,layer \
    --decoder-add-hyperadapters \
    --decoder-hyperadapter-lang-embedding-dim 64 \
    --decoder-hyperadapter-layer-embedding-dim 64 \
    --decoder-hyperadapter-bottleneck-dim 64 \
    --decoder-hyperadapter-hidden-dim 64 \
    --decoder-hyperadapter-num-hidden-layers 2 \
    --decoder-hyperadapter-generate-layernorm \
    --decoder-hyperadapter-language-embedding-tied \
    --decoder-hyperadapter-inputs src,tgt,layer \
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
    --user-dir ../model_configs \
    --update-freq 6 \
    --distributed-world-size 1 \
    --max-tokens 4096 \
    --lr 7.5e-4 \
    --restore-file ../checkpoints/$restore_from_dir/checkpoint_best.pt \
    --load-checkpoint-liberally \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-dataloader \
    --reset-optimizer \
    --num-workers 16 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    --memory-efficient-fp16

    restore_from_dir=$save_to_dir
done