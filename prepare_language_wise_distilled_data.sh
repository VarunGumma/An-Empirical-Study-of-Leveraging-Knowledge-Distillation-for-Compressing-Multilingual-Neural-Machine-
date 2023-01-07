#!/bin/bash

for langs in as,bn,or hi,pa,gu,mr ta,te,ml,kn; do 
    echo "working with ${langs}"
    ./prepare_distilled_data.sh \
    ../Downloads/data/v2_distilled_indic_en \
    ../Downloads/data/benchmarks \
    ../data_dir/v2_distilled_indic_en_language_family_LF1_bin/$langs \
    ../Downloads/indic-en \
    indic \
    en \
    $langs
done 
