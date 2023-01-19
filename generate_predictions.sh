#!/bin/bash
echo `date`

original_data_dir=$1
pred_data_dir=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
ckpt_dir=$6

for dir in `ls $original_data_dir`; do
    echo "working on ${dir}"
    for lang_pair in `ls $original_data_dir/$dir`; do
        mkdir -p $pred_data_dir/$dir/$lang_pair
        IFS='-' read -ra temp <<< $lang_pair

        if [[ "$src_lang" == en ]]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi

        if [ ! -f $pred_data_dir/$dir/$lang_pair/test.$src_lang ]; then
            echo "copying test files into ${pred_data_dir}/${dir}/${lang_pair}"
            cp $original_data_dir/$dir/$lang_pair/test.* $pred_data_dir/$dir/$lang_pair
        fi 

        bash joint_translate.sh \
            $pred_data_dir/$dir/$lang_pair/test.$src_lang \
            $pred_data_dir/$dir/$lang_pair/test_base.$tgt_lang \
            $src_lang \
            $tgt_lang \
            $ckpt_dir \
            $exp_dir
    done
done
