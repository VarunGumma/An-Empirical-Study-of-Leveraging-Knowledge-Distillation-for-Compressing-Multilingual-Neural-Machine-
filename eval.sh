#!/bin/bash

# Assign command line arguments to variables
flores_dir=$1
exp_dir=$2
src_lang=$3
tgt_lang=$4
shift 4

models=("$@")

# Remove previous results
rm -rf results/*
mkdir -p results

for model in "${models[@]}"; do

    save_path=results/$model
    mkdir -p $save_path

    for lang_pair in $flores_dir/*; do
        lang=$(basename "$lang_pair")

        # Set source and target languages
        IFS='-' read -ra temp <<< $lang
        if [[ $src_lang == "en" ]]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi

        predfile=$lang_pair/test.$tgt_lang.pred.$model
        srcfile=$lang_pair/test.$src_lang 
        tgtfile=$lang_pair/test.$tgt_lang
    
        # Translate and compute scores
        bash joint_translate.sh $srcfile $predfile $src_lang $tgt_lang $exp_dir $model
        bash compute_metrics.sh $predfile $tgtfile $tgt_lang > $save_path/$lang.json
        
        # comet-score \
        #     -s $srcfile \
        #     -t $predfile \
        #     -r $tgtfile \
        #     --num_workers 16 \
        #     --batch_size 256 \
        #     --gpus 1 \
        #     --quiet \
        #     --only_system \
        #     > $save_path/${lang}_comet.txt

        rm -rf $lang_pair/predfile.* $lang_pair/*.tok
        rm -rf $exp_dir/$model/*.out
    done
done

# Convert JSON files to CSV
echo -e "[INFO]\tConverting all JSON files to CSV"
python3 collate_metrics.py results ${models[@]}