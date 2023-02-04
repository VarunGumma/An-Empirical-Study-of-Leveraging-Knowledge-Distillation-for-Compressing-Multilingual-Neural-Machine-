#!/bin/bash

train_dir=$1
devtest_dir=$2
exp_dir=$3
src_lang=$4
tgt_lang=$5
languages_list=$6
reuse_bpe_vocab=$7
vocab_bpe_dir=$8
vocab_type=${9:-sep} # sep or joint
transliterate=${10:-true}
num_operations=${11:-32000}

echo `date`
rm -rf $exp_dir
mkdir -p $exp_dir

echo -e "[INFO]\tremoving overlap between train and devtest"
echo -e "[WARNING]\tthis operation will alter the train set you have passed!"
python3 scripts/remove_train_devtest_overlaps.py -t $train_dir -d $devtest_dir -l $languages_list

echo -e "[INFO]\tmerging devtest files"
rm -rf $devtest_dir/all
bash merge_benchmarks.sh $devtest_dir flores101_dataset $languages_list

echo -e "[INFO]\tcopying data"
IFS='+' read -ra langs <<< $languages_list
for lang in ${langs[@]}; do
    # copy only data you want to work with
    # saves space and is more efficient
    cp -r $train_dir/en-$lang $exp_dir
done
mkdir -p $exp_dir/devtest
cp -r $devtest_dir/all $exp_dir/devtest

echo -e "[INFO]\tpreparing data. It is recommened to use a multicore processor for this."
train_data_dir=$exp_dir
devtest_data_dir=$exp_dir/devtest/all

echo "Running experiment ${exp_dir} on ${src_lang} to ${tgt_lang}"
train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

mkdir -p $train_processed_dir
mkdir -p $devtest_processed_dir

IFS='+' read -ra langs <<< $languages_list

if [[ "$transliterate" == false ]]; then
	echo "skipping transliteration to Devnagiri"
fi

for lang in ${langs[@]}; do
	if [[ "$src_lang" == en ]]; then
		tgt_lang=$lang
	else
		src_lang=$lang
	fi
	echo "working on $src_lang"
	train_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	devtest_norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	mkdir -p $train_norm_dir
	mkdir -p $devtest_norm_dir

	# train preprocessing
	train_infname_src=$train_data_dir/en-${lang}/train.$src_lang
	train_infname_tgt=$train_data_dir/en-${lang}/train.$tgt_lang
	train_outfname_src=$train_norm_dir/train.$src_lang
	train_outfname_tgt=$train_norm_dir/train.$tgt_lang
	echo "Applying normalization and script conversion for train"
	# this is for preprocessing text and in for indic langs, we convert all scripts to devnagiri
	input_size=`python3 scripts/preprocess_translate.py $train_infname_src $train_outfname_src $src_lang $transliterate`
	input_size=`python3 scripts/preprocess_translate.py $train_infname_tgt $train_outfname_tgt $tgt_lang $transliterate`
	echo "Number of sentences in train: $input_size"
	# dev preprocessing
	dev_infname_src=$devtest_data_dir/en-${lang}/dev.$src_lang
	dev_infname_tgt=$devtest_data_dir/en-${lang}/dev.$tgt_lang
	dev_outfname_src=$devtest_norm_dir/dev.$src_lang
	dev_outfname_tgt=$devtest_norm_dir/dev.$tgt_lang
	echo "Applying normalization and script conversion for dev"
	input_size=`python3 scripts/preprocess_translate.py $dev_infname_src $dev_outfname_src $src_lang $transliterate`
	input_size=`python3 scripts/preprocess_translate.py $dev_infname_tgt $dev_outfname_tgt $tgt_lang $transliterate`
	echo "Number of sentences in dev: $input_size"
	# test preprocessing
	test_infname_src=$devtest_data_dir/en-${lang}/test.$src_lang
	test_infname_tgt=$devtest_data_dir/en-${lang}/test.$tgt_lang
	test_outfname_src=$devtest_norm_dir/test.$src_lang
	test_outfname_tgt=$devtest_norm_dir/test.$tgt_lang
	echo "Applying normalization and script conversion for test"
	input_size=`python3 scripts/preprocess_translate.py $test_infname_src $test_outfname_src $src_lang $transliterate`
	input_size=`python3 scripts/preprocess_translate.py $test_infname_tgt $test_outfname_tgt $tgt_lang $transliterate`
	echo "Number of sentences in test: $input_size"
done

# this concatenates lang pair data and creates text files to keep track of number of lines in each lang pair.
# this is imp as for joint training, we will merge all the lang pairs and the indivitual lang lines info
# would be required for adding specific lang tags later.

# the outputs of these scripts will  be text file like this:
# <lang1> <lang2> <number of lines>
# lang1-lang2 n1
# lang1-lang3 n2

python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'train'
python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'dev'
python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'test'

if [[ "$reuse_bpe_vocab" == false ]]; then
	echo "Learning bpe. This will take a very long time depending on the size of the dataset"
	if [[ "$vocab_type" == "sep" ]]; then
	    bash learn_single_bpe.sh $exp_dir $num_operations
	else 
	    bash learn_bpe.sh $exp_dir $num_operations
	fi
else
	echo "reusing old bpe"
	mkdir $exp_dir/final_bin
	# copy the old vocab
	cp -r $vocab_bpe_dir/vocab $exp_dir
	# copy the old dictionaries into the exp_dir/final_bin
	cp -r  $vocab_bpe_dir/final_bin/dict.* $exp_dir/final_bin
fi

echo "Applying bpe"
if [[ "$vocab_type" == "sep" ]]
then
    bash apply_single_bpe_traindevtest_notag.sh $exp_dir 'train'
	bash apply_single_bpe_traindevtest_notag.sh $exp_dir 'dev'
	bash apply_single_bpe_traindevtest_notag.sh $exp_dir 'test'
else 
    bash apply_bpe_traindevtest_notag.sh $exp_dir 'train'
	bash apply_bpe_traindevtest_notag.sh $exp_dir 'dev'
	bash apply_bpe_traindevtest_notag.sh $exp_dir 'test'
fi

mkdir -p $exp_dir/final
# # this is only required for joint training
# we apply language tags to the bpe segmented data
# if we are translating lang1 to lang2 then <lang1 line> will become __src__ <lang1> __tgt__ <lang2> <lang1 line>
echo "Adding language tags"
python3 scripts/add_joint_tags_translate.py $exp_dir 'train'
python3 scripts/add_joint_tags_translate.py $exp_dir 'dev'
python3 scripts/add_joint_tags_translate.py $exp_dir 'test'

# use cpu_count to get num_workers instead of setting it manually when running in different instances
num_workers=`python3 -c "import multiprocessing; print(multiprocessing.cpu_count())"`

# Binarize the training data for using with fairseq train
if [ "$reuse_bpe_vocab" == true ]; then
	echo "Binarizing using the existing vocab"
	fairseq-preprocess \
		--source-lang SRC \
		--target-lang TGT \
		--trainpref $exp_dir/final/train \
		--validpref $exp_dir/final/dev \
		--testpref $exp_dir/final/test \
		--destdir $exp_dir/final_bin \
		--srcdict $exp_dir/final_bin/dict.SRC.txt \
		--tgtdict $exp_dir/final_bin/dict.TGT.txt \
		--workers $num_workers \
		--thresholdtgt 5 \
		--thresholdsrc 5
else
	echo "Binarizing data"
	fairseq-preprocess \
		--source-lang SRC \
		--target-lang TGT \
		--trainpref $exp_dir/final/train \
		--validpref $exp_dir/final/dev \
		--testpref $exp_dir/final/test \
		--destdir $exp_dir/final_bin \
		--workers $num_workers \
		--thresholdtgt 5 \
		--thresholdsrc 5
fi

echo -e "[INFO]\tcleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe $exp_dir/devtest $exp_dir/final $exp_dir/data $exp_dir/norm $exp_dir/en-* $devtest_dir/all

echo -e "[INFO]\tcompleted!"