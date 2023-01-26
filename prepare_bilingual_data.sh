#!/bin/bash

# for lang in as bn gu hi kn ml mr or pa ta te; do
#     echo `date`
#     echo "working on ${lang}"
#     bash prepare_distilled_data.sh \
#     ../data_raw/v2 \
#     ../data_raw/benchmarks \
#     ../data_bin/bilingual_wo_transliteration/og/$lang \
#     indic \
#     en \
#     $lang \
#     false \
#     none \
#     false \
#     8192
# done


for lang in as bn gu hi kn ml mr or pa ta te; do
    echo `date`
    echo "working on ${lang}"
    bash prepare_distilled_data.sh \
    ../data_raw/v2_distilled_indic_en \
    ../data_raw/benchmarks \
    ../data_bin/bilingual_wo_transliteration/distilled/$lang \
    indic \
    en \
    $lang \
    true \
    ../data_bin/bilingual_wo_transliteration/og/$lang \
    false \
    8192
done