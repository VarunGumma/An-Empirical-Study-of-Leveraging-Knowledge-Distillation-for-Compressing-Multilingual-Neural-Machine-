#/bin/bash

for lang in as bn gu hi kn ml mr or pa te; do
    echo "working on ${lang}"
    bash prepare_distilled_data.sh \
    ../data_raw/v2_distilled_indic_en \
    ../data_raw/benchmarks \
    ../data_dir/bilingual/distilled/$lang \
    ../data_dir/v2_indic_en_bin \
    indic \
    en \
    $lang
done