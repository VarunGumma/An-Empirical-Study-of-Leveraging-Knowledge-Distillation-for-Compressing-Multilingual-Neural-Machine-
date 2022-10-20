#!/bin/bash

echo `date`
echo -e "[IMPORTANT]\tMAKE SURE THE VOCAB FOLDER IN THE EXPERIMENT DIRECTORY CONTAINS THE BPE CODES AND VOCABULARY OF THE ORIGINAL INDICTRANS MODEL!"

echo -e "[INFO]\tremoving overlap between train and devtest"
cd indicTrans/scripts
python3 remove_train_devtest_overlaps.py ../../samanantar_data ../../benchmarks
cd ../..

echo -e "[INFO]\tmerging devtest files"
cd benchmarks
python3 merge_benchmarks.py
cd ..

echo -e "[INFO]\tcopying data"
cp -r samanantar_data/* indic-en-exp
mkdir -p indic-en-exp/devtest
cp -r benchmarks/all indic-en-exp/devtest/

echo -e "[INFO]\tpreparing data. It is recommened you use a multicore processor for this."
cd indicTrans
bash prepare_data_joint_training.sh ../indic-en-exp indic en
cd ..

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
cd indic-en-exp
rm README.md
rm -rf bpe devtest final data norm en-bn en-hi en-ml en-or en-ta en-as en-gu en-kn en-mr en-pa en-te