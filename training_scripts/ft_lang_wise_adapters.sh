#!/bin/bash

data_dir=$1
model_name=$2
restore_from_dir=$3

for lang in as bn gu hi kn ml mr or pa ta te; do
    echo `date`
    echo -e "\n[INFO]\tfinetuning on ${lang}"

    if [[ "$lang" == as ]]; then
        warmup=1000
        update_freq=1
    elif [[ "$lang" == gu ]]; then
        warmup=2000 # for xx-en
        # warmup=4000 # for en-xx
        update_freq=4
    elif [[ "$lang" == or ]]; then
        warmup=1600 # for xx-en
        # warmup=2000 # for en-xx
        update_freq=4
    else
        warmup=4000
        update_freq=4
    fi

    fairseq-train ${data_dir}/$lang/final_bin \
        --max-source-positions 210 \
        --max-target-positions 210 \
        --max-update 1000000 \
        --save-interval 1 \
        --save-interval-updates $warmup \
        --arch transformer_${model_name} \
        --encoder-add-adapters \
        --encoder-adapter-reduction-factor 2 \
        --encoder-adapter-ids as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
        --encoder-train-adapter $lang \
        --decoder-add-adapters \
        --decoder-adapter-reduction-factor 2 \
        --decoder-adapter-ids as,bn,gu,hi,kn,ml,mr,or,pa,ta,te \
        --decoder-train-adapter $lang \
        --adapter-activation-fn gelu \
        --criterion label_smoothed_cross_entropy \
        --source-lang SRC \
        --target-lang TGT \
        --lr-scheduler inverse_sqrt \
        --label-smoothing 0.1 \
        --optimizer adam \
        --adam-betas "(0.9, 0.98)" \
        --clip-norm 1.0 \
        --warmup-init-lr 1e-07 \
        --warmup-updates $warmup \
        --dropout 0.2 \
        --restore-file ${restore_from_dir}/checkpoint_best.pt \
        --save-dir ${data_dir}/$lang/${model_name} \
        --no-epoch-checkpoints \
        --keep-interval-updates 1 \
        --patience 5 \
        --skip-invalid-size-inputs-valid-test \
        --update-freq $update_freq \
        --distributed-world-size 1 \
        --max-tokens 2048 \
        --lr 1e-3 \
        --load-checkpoint-liberally \
        --reset-lr-scheduler \
        --reset-meters \
        --reset-dataloader \
        --reset-optimizer \
        --num-workers 24 \
        --eval-bleu \
        --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
        --eval-bleu-detok moses \
        --eval-bleu-remove-bpe \
        --maximize-best-checkpoint-metric \
        --best-checkpoint-metric bleu \
        --memory-efficient-fp16

    restore_from_dir=${data_dir}/$lang/${model_name}
    echo "====================================================================================="
done
