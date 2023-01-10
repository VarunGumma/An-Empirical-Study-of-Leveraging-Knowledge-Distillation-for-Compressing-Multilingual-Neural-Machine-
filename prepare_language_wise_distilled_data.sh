#!/bin/bash

for langs in as+bn+or hi+pa+gu+mr ta+te+kn+ml; do 
    echo "working with ${langs}"
    bash prepare_distilled_data.sh \
    ../Downloads/data/v2_distilled_indic_en \
    ../Downloads/data/benchmarks \
    ../data_dir/temp/$langs \
    ../Downloads/indic-en \
    indic \
    en \
    $langs
done 
