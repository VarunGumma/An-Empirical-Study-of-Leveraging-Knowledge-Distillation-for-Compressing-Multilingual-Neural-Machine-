#!/bin/bash

train_dir=$1
exp_dir=$2
src_lang=$3
tgt_lang=$4
languages_list=$5
vocab_bpe_dir=$6
vocab_type=${7:-sep} # sep or joint
transliterate=${8:-true}

echo `date`
rm -rf $exp_dir
mkdir -p $exp_dir

echo -e "[INFO]\tcopying data"
IFS='+' read -ra langs <<< $languages_list

for lang in ${langs[@]}; do
    cp -r $train_dir/en-$lang $exp_dir
done

echo -e "[INFO]\tpreparing data. It is recommened to use a multicore processor for this."
train_data_dir=$exp_dir
train_processed_dir=$exp_dir/data
mkdir -p $train_processed_dir

IFS='+' read -ra langs <<< $languages_list
if [[ "$transliterate" == false ]]; then
	echo -e "[INFO]\tskipping transliteration to Devanagiri"
fi

for lang in ${langs[@]}; do
	if [[ "$src_lang" == en ]]; then
		tgt_lang=$lang
	else
		src_lang=$lang
	fi
	echo -e "[INFO]\tworking on $src_lang-$tgt_lang"
	train_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	mkdir -p $train_norm_dir

	train_infname_src=$train_data_dir/en-${lang}/train.$src_lang
	train_infname_tgt=$train_data_dir/en-${lang}/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang
	echo -e "[INFO]\tApplying normalization and script conversion for train"
	# this is for preprocessing text and in for indic langs, we convert all scripts to devnagiri
	input_size=`python3 scripts/preprocess_translate.py $train_infname_src $train_outfname_src $src_lang $transliterate`
	input_size=`python3 scripts/preprocess_translate.py $train_infname_tgt $train_outfname_tgt $tgt_lang $transliterate`
	echo "Number of sentences in train: $input_size"
done

python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'train'

echo -e "[INFO]\tReusing old bpe"
mkdir $exp_dir/final_bin
cp -r $vocab_bpe_dir/vocab $exp_dir
cp -r $vocab_bpe_dir/final_bin/dict.* $exp_dir/final_bin

echo -e "[INFO]\tApplying bpe"
if [[ "$vocab_type" == "sep" ]]
then
    bash apply_single_bpe_traindevtest_notag.sh $exp_dir 'train'
else 
    bash apply_bpe_traindevtest_notag.sh $exp_dir 'train'
fi

echo -e "[INFO]\tAdding language tags"
mkdir -p $exp_dir/final
python3 scripts/add_joint_tags_translate.py $exp_dir 'train'

num_workers=`python3 -c "import multiprocessing; print(multiprocessing.cpu_count())"`

fairseq-preprocess \
--source-lang SRC \
--target-lang TGT \
--trainpref $exp_dir/final/train \
--destdir $exp_dir/final_bin \
--srcdict $exp_dir/final_bin/dict.SRC.txt \
--tgtdict $exp_dir/final_bin/dict.TGT.txt \
--workers $num_workers \
--thresholdtgt 5 \
--thresholdsrc 5

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe $exp_dir/devtest $exp_dir/final $exp_dir/data $exp_dir/norm $exp_dir/en-* $devtest_dir/all

echo -e "[INFO]\tcompleted!"