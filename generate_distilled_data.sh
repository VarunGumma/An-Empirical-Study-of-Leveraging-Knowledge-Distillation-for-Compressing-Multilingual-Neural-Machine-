#!/bin/bash
echo `date`

original_data_dir=$1
distilled_data_dir=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5

echo "removing previous distillation data directory (if it exists)"
rm -rf $distilled_data_dir

for dir in `ls $original_data_dir`; do
    echo "working on ${dir}"
    mkdir -p $distilled_data_dir/$dir
    IFS='-' read -ra temp <<< $dir
    if [ $src_lang == en ]; then
		tgt_lang=${temp[1]}
	else
		src_lang=${temp[1]}
	fi
    cp $original_data_dir/$dir/train.$src_lang $distilled_data_dir/$dir
    bash joint_translate.sh $distilled_data_dir/$dir/train.$src_lang $distilled_data_dir/$dir/train.$tgt_lang $src_lang $tgt_lang $exp_dir/model $exp_dir
done
