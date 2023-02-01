#!/bin/bash

for lang in as bn gu hi kn ml mr or pa ta te; do
    echo "working on ${lang}"
    bash prepare_train_set.sh \
    ../Downloads/data_raw/v2 \
    ../data_bin/train_bin/en-$lang \
    indic \
    en \
    $lang \
    ../data_bin/v2_indic_en_bin
done