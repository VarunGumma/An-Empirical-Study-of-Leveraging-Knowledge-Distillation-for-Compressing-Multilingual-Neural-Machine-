#!/bin/bash

for langs in ta+kn+ml; do 
    echo "working with ${langs}"
    bash prepare_distilled_data.sh \
    ../data_raw/v2_distilled_indic_en \
    ../data_dir/benchmarks \
    ../data_dir/temp/$langs \
    ../data_dir/v2_indic_en_bin \
    indic \
    en \
    $langs
done 
