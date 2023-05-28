#!/bin/bash

devtest_dir=$1
exp_dir=$2
model=$3
src_lang=$4
tgt_lang=$5
save_path=$6/$model

python3 ctranslate2_inference/translate.py \
    --devtest-dir $devtest_dir \
    --exp-dir $exp_dir \
    --model $model \
    --source-lang $src_lang \
    --target-lang $tgt_lang


for lang_pair in $devtest_dir/*; do
    lang=$(basename "$lang_pair")
    IFS='-' read -ra temp <<< $lang
    if [[ $src_lang == "en" ]]; then
        t=${temp[1]}
    else
        s=${temp[1]}
    fi

    bash compute_metrics.sh test.$t.pred.$model test.$t $t > $save_path/$lang.json
done

python3 collate_metrics.py results $model
