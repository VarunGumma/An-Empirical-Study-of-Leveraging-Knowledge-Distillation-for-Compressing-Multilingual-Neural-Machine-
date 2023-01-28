#!/bin/bash
echo `date`

devtest_base_dir=$1
ckpt_base_dir=$2
exp_dir=$3
src_lang=$4
tgt_lang=$5
transliterate=$6
shift 6

models=("$@")

rm -rf results/*

for model in "${models[@]}"; do
    for dir in `ls $devtest_base_dir`; do
        echo -e "\n<<<<<<<<<< ${dir} >>>>>>>>>>"
        for lang_pair in `ls $devtest_base_dir/$dir`; do

            echo "working on ${lang_pair}"
            path=$devtest_base_dir/$dir/$lang_pair
            save_path="results/${model}/${dir}/${lang_pair}"

            IFS='-' read -ra temp <<< $lang_pair

            mkdir -p $save_path

            if [[ "$src_lang" == en ]]; then
		        tgt_lang=${temp[1]}
	        else
		        src_lang=${temp[1]}
	        fi
            
            if [[ -f $path/dev.$src_lang ]]; then
                bash joint_translate.sh $path/dev.$src_lang $path/outfile.$tgt_lang $src_lang $tgt_lang $ckpt_base_dir/$model $exp_dir $transliterate
                bash compute_bleu.sh $path/outfile.$tgt_lang $path/dev.$tgt_lang $src_lang $tgt_lang > $save_path/${temp[1]}.json
            fi

        done 
    done
done


echo -e "[INFO]\tconverting all json files to csv"
python3 json_to_csv.py "$@"
