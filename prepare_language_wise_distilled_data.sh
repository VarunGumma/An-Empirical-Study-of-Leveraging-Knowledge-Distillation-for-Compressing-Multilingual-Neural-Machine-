#!/bin/bash

for lang in as bn gu hi kn ml mr or pa ta te; do 
    bash prepare_distilled_data.sh v2_distilled_indic_en_HQ benchmarks v2_distilled_indic_en_language_wise_HQ_bin/$lang checkpoints/indic-en $lang en
done 