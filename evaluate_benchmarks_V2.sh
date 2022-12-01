#!/bin/bash
echo `date`
devtest_base_dir=$1
ckpt_base_dir=$2
src_lang=$3
tgt_lang=$4

for dir in flores101_dataset wat2021-devtest wat2020-devtest; do
    echo ">>>>> ${dir}"
    for lang_pair in `ls $devtest_base_dir/$dir`; do
        path=$devtest_base_dir/$dir/$lang_pair
        IFS='-' read -ra temp <<< $lang_pair
        if [ $src_lang == en ]; then
            tgt_lang=${temp[1]}
        else
            src_lang=${temp[1]}
        fi
        bash joint_translate_V2.sh $path/dev.$src_lang $path/outfile.$tgt_lang $src_lang $tgt_lang $ckpt_base_dir/model $ckpt_base_dir $path/indic_en_encoder_states
    done 
    echo -e "<<<<< ${dir}\n"
done