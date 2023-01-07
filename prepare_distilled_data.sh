#!/bin/bash

train_dir=$1
devtest_dir=$2
exp_dir=$3
vocab_bpe_dir=$4
src_lang=$5
tgt_lang=$6
languages_list=$7

echo `date`
echo -e "[IMPORTANT]\tMAKE SURE THE VOCAB FOLDER IN THE EXPERIMENT DIRECTORY CONTAINS THE BPE CODES AND VOCABULARY OF THE ORIGINAL INDICTRANS MODEL!"

rm -rf $exp_dir
mkdir -p $exp_dir

echo -e "[INFO]\tremoving overlap between train and devtest"

python3 scripts/remove_train_devtest_overlaps.py -t $train_dir -d $devtest_dir -l $languages_list

rm -rf $devtest_dir/all
echo -e "[INFO]\tmerging devtest files"
bash merge_benchmarks.sh $devtest_dir $languages_list flores101_dataset

echo -e "[INFO]\tcopying data"
IFS=',' read -ra langs <<< $languages_list
for lang in ${langs[@]}; do
    # copy only data you want to work with
    # saves space and is more efficient
    cp -r $train_dir/en-$lang $exp_dir
done
mkdir -p $exp_dir/devtest
cp -r $devtest_dir/all $exp_dir/devtest

echo -e "[INFO]\tpreparing data. It is recommened to use a multicore processor or a GPU for this."
bash prepare_data_joint_training.sh $exp_dir $src_lang $tgt_lang $languages_list sep $exp_dir $exp_dir/devtest/all true $vocab_bpe_dir

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe $exp_dir/devtest $exp_dir/final $exp_dir/data $exp_dir/norm $exp_dir/en-* $devtest_dir/all

echo -e "[INFO]\tcompleted!"
