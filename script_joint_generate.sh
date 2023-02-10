#!/bin/bash

for lang in as bn gu hi kn ml mr or pa ta te; do 
    echo "working on ${lang}"
    bash joint_generate.sh \
    ../data_bin/train_bin/en-$lang \
    ../data_raw/v2/en-$lang/distil.en \
    $lang \
    en \
    checkpoints/it
done 