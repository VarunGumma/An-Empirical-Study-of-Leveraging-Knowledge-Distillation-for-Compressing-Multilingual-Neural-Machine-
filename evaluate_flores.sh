#!/bin/bash
echo `date`

flores_dir=$1
exp_dir=$2
src_lang=$3
tgt_lang=$4
transliterate=$5
shift 5

models=("$@")

rm -rf results/*

for model in "${models[@]}"; do
    for lang_pair in `ls $flores_dir; do

        echo "working on ${lang_pair}"
        path="${flores_dir}/${lang_pair}"
        save_path="results/${model}/flores/${lang_pair}"

        IFS='-' read -ra temp <<< $lang_pair

        mkdir -p $save_path

        if [[ "$src_lang" == en ]]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi
        
        if [[ -f $path/test.$src_lang ]]; then
            bash joint_translate.sh $path/test.$src_lang $path/outfile.$tgt_lang $src_lang $tgt_lang checkpoints/$model $exp_dir $transliterate
            bash compute_bleu.sh $path/outfile.$tgt_lang $path/test.$tgt_lang $src_lang $tgt_lang > $save_path/${temp[1]}.json
        fi
    done 
done


echo -e "[INFO]\tconverting all json files to csv"
python3 json_to_csv.py "$@"
