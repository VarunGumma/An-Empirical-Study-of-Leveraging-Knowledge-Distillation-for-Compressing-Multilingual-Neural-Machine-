#!/bin/bash

for lang in as bn gu hi kn ml mr or pa ta te; do 
    echo "working on ${lang}"
    bash joint_generate.sh \
    ../data_hq_bin/train_hq2_bin/en-$lang \
    ../v2_hq2/en-$lang/distil.en \
    $lang \
    en \
    checkpoints/it_hq2
done 