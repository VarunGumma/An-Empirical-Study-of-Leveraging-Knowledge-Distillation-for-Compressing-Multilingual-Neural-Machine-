#!/bin/bash

for lang in as bn gu hi kn ml mr or pa te; do
    echo -e "\n${lang}:"
    bash compute_bleu.sh data_raw/v2_distilled_indic_en/en-$lang/train.en data_raw/v2/en-$lang/train.en $lang en
done