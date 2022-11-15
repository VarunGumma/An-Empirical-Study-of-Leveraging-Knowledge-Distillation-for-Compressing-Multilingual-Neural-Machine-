#!/bin/bash

train_dir=$1
devtest_dir=$2
exp_dir=$3

echo `date`
echo -e "[IMPORTANT]\tMAKE SURE THE VOCAB FOLDER IN THE EXPERIMENT DIRECTORY CONTAINS THE BPE CODES AND VOCABULARY OF THE ORIGINAL INDICTRANS MODEL!"

echo -e "[INFO]\tremoving overlap between train and devtest"
cd indicTrans/scripts
python3 remove_train_devtest_overlaps.py ../../$train_dir ../../$devtest_dir
cd ../..

if [ ! -d "../../$devtest_dir/all" ] 
then
    echo -e "[INFO]\tmerging devtest files"
    cd $devtest_dir
    python3 merge_benchmarks.py
    cd ..
fi 

echo -e "[INFO]\tcopying data"
cp -r $train_dir/* $exp_dir
mkdir -p $exp_dir/devtest
cp -r $devtest_dir/all $exp_dir/devtest/

echo -e "[INFO]\tpreparing data. It is recommened you use a multicore processor for this."
cd indicTrans
bash prepare_data_joint_training.sh ../$exp_dir indic en sep ../$exp_dir ../$exp_dir/devtest/all true ../indic-en
cd ..

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
cd $exp_dir
rm README.md
rm -rf bpe devtest final data norm en-bn en-hi en-ml en-or en-ta en-as en-gu en-kn en-mr en-pa en-te

echo -e "[INFO]\tcompleted!"