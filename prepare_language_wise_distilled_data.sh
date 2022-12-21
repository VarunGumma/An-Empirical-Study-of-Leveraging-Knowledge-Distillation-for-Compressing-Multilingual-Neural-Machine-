#!/bin/bash

for langs in hi-pa-gu-mr; do 
    bash prepare_distilled_data.sh \
    ../Downloads/data/v2_distilled_indic_en \
    ../data_dir/benchmarks \
    ../data_dir/v2_distilled_indic_en_language_family_LF1_bin_temp/$langs \
    checkpoints/indic-en \
    indic \
    en \
    $langs
done 