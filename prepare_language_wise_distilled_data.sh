#!/bin/bash

for langs in hi:pa:gu:mr as:bn:or ta:te:ml:kn; do 
    bash prepare_distilled_data.sh \
    ../Downloads/data/v2_distilled_indic_en \
    ../data_dir/benchmarks \
    ../data_dir/v2_distilled_indic_en_language_family_v1_bin/$langs \
    checkpoints/indic-en \
    indic \
    en \
    $langs
done 