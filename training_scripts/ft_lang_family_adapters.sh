#!/bin/bash

data_dir=$1
restore_from_dir=$2

for lang in as+bn+or gu+hi+mr+pa kn+ml+ta+te; do
    echo `date`
    echo -e "\n[INFO]\tfinetuning on ${lang}"

    fairseq-train $data_dir/$lang/final_bin \
    --max-source-positions 210 \
    --max-target-positions 210 \
    --max-update 1000000 \
    --arch transformer_base \
    --encoder-add-adapters \
    --encoder-adapter-bottleneck-dim 256 \
    --encoder-adapter-langs as+bn+or,gu+hi+mr+pa,kn+ml+ta+te \
    --encoder-finetune-adapter $lang \
    --encoder-adapter-type houlsby \
    --decoder-add-adapters \
    --decoder-adapter-bottleneck-dim 256 \
    --decoder-adapter-langs as+bn+or,gu+hi+mr+pa,kn+ml+ta+te \
    --decoder-finetune-adapter $lang \
    --decoder-adapter-type houlsby \
    --adapter-activation-fn gelu \
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
    --warmup-updates 4000 \
    --dropout 0.2 \
    --restore-file ${restore_from_dir}/checkpoint_best.pt \
    --save-dir ${data_dir}/$lang \
    --save-interval 1 \
    --save-interval-updates 4000 \
    --no-epoch-checkpoints \
    --keep-interval-updates 1 \
    --patience 5 \
    --skip-invalid-size-inputs-valid-test \
    --update-freq 3 \
    --distributed-world-size 1 \
    --max-tokens 8192 \
    --lr 1e-3 \
    --load-checkpoint-liberally \
    --reset-lr-scheduler \
    --reset-meters \
    --reset-dataloader \
    --reset-optimizer \
    --num-workers 15 \
    --eval-bleu \
    --eval-bleu-args '{"beam": 5, "lenpen": 1.0, "max_len_a": 1.2, "max_len_b": 10}' \
    --eval-bleu-detok moses \
    --eval-bleu-remove-bpe \
    --maximize-best-checkpoint-metric \
    --best-checkpoint-metric bleu \
    --memory-efficient-fp16 \
    --user-dir ../model_configs
    
    restore_from_dir=${data_dir}/$lang
    echo "====================================================================================="
done
