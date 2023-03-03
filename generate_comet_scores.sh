#!/bin/bash

base_path=$1

comet_model="wmt20-comet-qe-da-v2"

echo "using ${comet_model}"

for lang_pair in $(ls $base_path | sort); do
    full_path="${base_path}/${lang_pair}"
    echo $full_path
    src=$(echo $lang_pair | cut -d'-' -f1)
    tgt=$(echo $lang_pair | cut -d'-' -f2)
    
    mv "${full_path}/scores.txt" "${full_path}/labse.txt"
    
    cmd="comet-score -s ${full_path}/train.${src} -t ${full_path}/train.${tgt} --num_workers 16 --batch_size 256 --seed_everything 2023 --model ${comet_model}"

    out=$(eval $cmd | sed '$d')
    scores=$(echo "$out" | awk -F"\t" '{print $NF}' | awk -F": " '{print $NF}')
    
    printf '%s\n' "${scores[@]}" > "${full_path}/comet.txt"
done