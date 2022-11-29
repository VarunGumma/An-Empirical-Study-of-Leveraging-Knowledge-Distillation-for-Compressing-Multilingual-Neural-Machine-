#!/bin/bash

train_dir=$1
devtest_dir=$2
exp_dir=$3
vocab_bpe_dir=$4

echo `date`
echo -e "[IMPORTANT]\tMAKE SURE THE VOCAB FOLDER IN THE EXPERIMENT DIRECTORY CONTAINS THE BPE CODES AND VOCABULARY OF THE ORIGINAL INDICTRANS MODEL!"

mkdir $exp_dir

echo -e "[INFO]\tremoving overlap between train and devtest"
python3 scripts/remove_train_devtest_overlaps.py $train_dir $devtest_dir

if [ ! -d $devtest_dir/all ]; then
    echo -e "[INFO]\tmerging devtest files"
    bash merge_benchmarks.sh $devtest_dir
fi 

echo -e "[INFO]\tcopying data"
cp -r $train_dir/* $exp_dir
mkdir -p $exp_dir/devtest
cp -r $devtest_dir/all $exp_dir/devtest/

echo -e "[INFO]\tpreparing data. It is recommened to use a multicore processor or a GPU for this."
bash prepare_data_joint_training.sh $exp_dir indic en sep $exp_dir $exp_dir/devtest/all true $vocab_bpe_dir

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe \
    $exp_dir/devtest \
    $exp_dir/final \
    $exp_dir/data \
    $exp_dir/norm \
    $exp_dir/en-bn \
    $exp_dir/en-hi \
    $exp_dir/en-ml \
    $exp_dir/en-or \
    $exp_dir/en-ta \
    $exp_dir/en-as \
    $exp_dir/en-gu \
    $exp_dir/en-kn \
    $exp_dir/en-mr \
    $exp_dir/en-pa \
    $exp_dir/en-te

echo -e "[INFO]\tcompleted!"
