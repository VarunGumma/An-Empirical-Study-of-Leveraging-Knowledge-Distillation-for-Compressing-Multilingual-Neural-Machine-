#!/bin/bash
echo `date`

devtest_base_dir=$1
exp_dir=$2
src_lang=$3
tgt_lang=$4
transliterate=$5

rm -rf results/*

for dir in `ls $devtest_base_dir`; do
    echo -e "\n<<<<<<<<<< ${dir} >>>>>>>>>>"
    for lang_pair in `ls $devtest_base_dir/$dir`; do

        echo "working on ${lang_pair}"
        path="${devtest_base_dir}/${dir}/${lang_pair}"
        save_path="results/${ckpt_base_dir}/${dir}/${lang_pair}"

        IFS='-' read -ra temp <<< $lang_pair

        mkdir -p $save_path

        if [[ $transliterate == true ]]; then
            model=${temp[1]}
        else
            model=${temp[1]}_wo_transliteration 
        fi

        if [[ "$src_lang" == en ]]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi
        
        if [[ -f $path/test.$src_lang ]]; then
            bash joint_translate.sh $path/test.$src_lang $path/outfile.$tgt_lang $src_lang $tgt_lang checkpoints/$ckpt_base_dir/$model $exp_dir $transliterate
            bash compute_bleu.sh $path/outfile.$tgt_lang $path/test.$tgt_lang $src_lang $tgt_lang > $save_path/${temp[1]}.json
        fi
    done 
done

echo -e "[INFO]\tconverting all json files to csv"
python3 json_to_csv.py $ckpt_base_dir
