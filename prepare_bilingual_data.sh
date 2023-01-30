#!/bin/bash

for lang in as bn gu hi kn ml mr or pa ta te; do
    echo `date`
    bash prepare_distilled_data.sh \
    ../data_raw/v2 \
    ../data_raw/benchmarks \
    ../data_bin/bilingual/og/$lang \
    indic \
    en \
    $lang \
    false \
    none \
    true \
    8192

    echo `date`
    bash prepare_distilled_data.sh \
    ../data_raw/v2_distilled_indic_en \
    ../data_raw/benchmarks \
    ../data_bin/bilingual/distilled/$lang \
    indic \
    en \
    $lang \
    true \
    ../data_bin/bilingual/og/$lang \
    true \
    8192
done