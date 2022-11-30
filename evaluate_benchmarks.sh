#!/bin/bash
echo `date`

devtest_dir=$1
ckpt_base_dir=$2
exp_dir=$3
shift 3

allArgs=("$@")

for ext in "${allArgs[@]}"; do
    for dir in $devtest_dir/*; do
        echo ">>>>> ${dir}"
        for path in $dir/*; do
            IFS='-' read -ra temp1 <<< $path
            IFS='/' read -ra temp2 <<< $path
            lang=${temp1[1]}
            ds=${temp2[-2]}
            bash joint_translate.sh $path/test.$lang $path/outfile.en $lang en $ckpt_base_dir/$ext $exp_dir
            output=$(bash compute_bleu.sh $path/outfile.en $path/test.en $lang en)
            echo -e "${ds} - ${lang}: ${output}\n" >> results/$ext.txt
        done 
        echo -e "<<<<< ${dir}\n"
    done 
done

echo -e "[INFO]\tconverting all txt files to csv"
python3 convert_txt_to_csv.py "$@"
