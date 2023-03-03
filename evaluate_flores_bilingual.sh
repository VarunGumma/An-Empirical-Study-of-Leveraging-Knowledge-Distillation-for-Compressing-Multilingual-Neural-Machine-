#!/bin/bash

# Print current date
echo $(date)

# Assign command line arguments to variables
flores_dir=$1
ckpt_base_dir=$2
exp_dir=$3
src_lang=$4
tgt_lang=$5
transliterate=${6:-true}

# Remove previous results
rm -rf results/*
mkdir -p results

# Loop over language pairs
for lang_pair in "$flores_dir"/*; do
    echo "Working on - $lang_pair"

    # Extract source and target languages
    IFS='-' read -r src_lang_tgt_lang <<< $(basename "$lang_pair")
    src_lang=${src_lang_tgt_lang%-*}
    tgt_lang=${src_lang_tgt_lang#*-}

    # Check if test files and model directory exist
    if [[ ! -f "$lang_pair/test.$src_lang" ]] || [[ ! -d "checkpoints/$ckpt_base_dir" ]]; then
        continue
    fi

    # Set paths
    path="$lang_pair"
    model_path="checkpoints/$ckpt_base_dir/$tgt_lang"
    save_path="results/$ckpt_base_dir/flores/$src_lang_tgt_lang"
    mkdir -p "$save_path"

    # Translate and compute BLEU score
    bash joint_translate.sh "$path/test.$src_lang" "$path/outfile.$tgt_lang" "$src_lang" "$tgt_lang" "$model_path" "$exp_dir/$ckpt_base_dir/$tgt_lang" "$transliterate"
    if [[ -f "$path/outfile.$tgt_lang" ]]; then
        bash compute_bleu.sh "$path/outfile.$tgt_lang" "$path/test.$tgt_lang" "$src_lang" "$tgt_lang" > "$save_path/$tgt_lang.json"
    else
        echo "Translation failed for $lang_pair"
    fi
done

# Convert JSON files to CSV
echo -e "[INFO]\tConverting all JSON files to CSV"
python3 json_to_csv.py "$ckpt_base_dir"
