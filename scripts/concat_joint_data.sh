#!/bin/bash

# Get command line arguments
in_dir=$1
out_dir=$2
src_lang=$3
tgt_lang=$4
languages_list=$5
split="$6"

out_src_lang="SRC"
out_tgt_lang="TGT"

# Create the output directory if it doesn't exist
mkdir -p $out_dir

# Split the language list into an array
IFS='+' read -ra INDIC_LANGS <<< $languages_list

# Create the language pair list
lang_pair_list=()
for lang in ${INDIC_LANGS[@]}; do
    if [ "$src_lang" = "en" ]; then
        lang_pair_list+=("en-$lang")
    else
        lang_pair_list+=("$lang-en")
    fi
done

# Concatenate files
for pair in ${lang_pair_list[@]}; do
    echo "$pair"
    in_src_fname="${in_dir}/${pair}/${split}.${src_lang}"
    in_trg_fname="${in_dir}/${pair}/${split}.${tgt_lang}"
    out_src_fname="${out_dir}/${split}.${out_src_lang}"
    out_trg_fname="${out_dir}/${split}.${out_tgt_lang}"

    if [ -e $in_src_fname ] && [ -e $in_trg_fname ]; then
        cat $in_src_fname >> $out_src_fname
        cat $in_trg_fname >> $out_trg_fname
    fi
done

# Generate stats
for pair in ${lang_pair_list[@]}; do
    in_src_fname="${in_dir}/${pair}/${split}.${src_lang}"

    if [ -e $in_src_fname ]; then
        num_lines=$(grep -c '.' "$in_src_fname")
        echo -e "${src_lang}\t${tgt_lang}\t${num_lines}" >> "${out_dir}/${split}_lang_pairs.txt"
    fi
done
