#!/bin/bash
echo `date`

devtest_base_dir=$1
ckpt_base_dir=$2
exp_dir=$3
src_lang=$4
tgt_lang=$5
shift 5

allArgs=("$@")

rm -rf results/*

for ext in "${allArgs[@]}"; do
    for dir in `ls $devtest_base_dir`; do
        echo ">>>>> ${dir}"
        for lang_pair in `ls $devtest_base_dir/$dir`; do
            echo "woring on ${lang_pair}"
            path=$devtest_base_dir/$dir/$lang_pair
            IFS='-' read -ra temp <<< $lang_pair
            if [ $src_lang == en ]; then
		        tgt_lang=${temp[1]}
	        else
		        src_lang=${temp[1]}
	        fi
            bash joint_translate.sh $path/test.$src_lang $path/outfile.$tgt_lang $src_lang $tgt_lang $ckpt_base_dir/$ext $exp_dir
            output=$(bash compute_bleu.sh $path/outfile.$tgt_lang $path/test.$tgt_lang $src_lang $tgt_lang)
            echo -e "${dir} - ${temp[1]}: ${output}\n" >> results/$ext.txt
        done 
        echo -e "<<<<< ${dir}\n"
    done
done

echo -e "[INFO]\tconverting all txt files to csv"
python3 convert_txt_to_csv.py "$@"
