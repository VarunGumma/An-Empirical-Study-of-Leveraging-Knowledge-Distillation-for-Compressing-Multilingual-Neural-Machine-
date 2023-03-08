#!/bin/bash

# Assign command line arguments to variables
flores_dir=$1
exp_dir=$2
src_lang=$3
tgt_lang=$4
transliterate=$5
shift 5
models=("$@")

# Remove previous results
rm -rf results/*
mkdir -p results

for model in "${models[@]}"; do
    for lang_pair in "$flores_dir"/*; do
        lang=$(basename "$lang_pair")

        # Set source and target languages
        IFS='-' read -ra temp <<< $lang
        if [[ "$src_lang" == "en" ]]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi

        # Check if test files exist
        if [[ ! -f "$lang_pair/test.$src_lang" ]]; then
            echo "Test files for $lang_pair not found"
            continue
        fi

        # Set paths
        outfile="$lang_pair/outfile.$tgt_lang"
        save_path="results/$model"
        mkdir -p "$save_path"

        # Translate
        bash joint_translate.sh "$lang_pair/test.$src_lang" "$outfile" "$src_lang" "$tgt_lang" "checkpoints/$model" "$exp_dir" "$transliterate"

        # Compute BLEU score
        if [[ -f "$outfile" ]]; then
            bash compute_bleu.sh "$outfile" "$lang_pair/test.$tgt_lang" "$src_lang" "$tgt_lang" > "$save_path/$lang.json"
        else
            echo "Translation failed for $lang_pair"
        fi
    done
done

# Convert JSON files to CSV
echo -e "[INFO]\tConverting all JSON files to CSV"
python3 json_to_csv.py "${models[@]}"