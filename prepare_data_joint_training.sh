#/bin/bash

exp_dir=$1
src_langs=$2
tgt_lang=$3
languages_list=$4
vocab_type=${5:-"sep"} # sep or joint
train_data_dir=${6:-"$exp_dir"}
devtest_data_dir=${7:-"$exp_dir/devtest/all"}
reuse_bpe_vocab=${8:-false}
vocab_bpe_dir=${9-"none"}

echo "Running experiment ${exp_dir} on ${src_langs} to ${tgt_lang}"

train_processed_dir=$exp_dir/data
devtest_processed_dir=$exp_dir/data

mkdir -p $train_processed_dir
mkdir -p $devtest_processed_dir

IFS=',' read -ra langs <<< $languages_list

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
	input_size=`python3 scripts/preprocess_translate.py $train_infname_src $train_outfname_src $src_lang true`
	input_size=`python3 scripts/preprocess_translate.py $train_infname_tgt $train_outfname_tgt $tgt_lang true`
	echo "Number of sentences in train: $input_size"
	# dev preprocessing
	dev_infname_src=$devtest_data_dir/en-${lang}/dev.$src_lang
	dev_infname_tgt=$devtest_data_dir/en-${lang}/dev.$tgt_lang
	dev_outfname_src=$devtest_norm_dir/dev.$src_lang
	dev_outfname_tgt=$devtest_norm_dir/dev.$tgt_lang
	echo "Applying normalization and script conversion for dev"
	input_size=`python3 scripts/preprocess_translate.py $dev_infname_src $dev_outfname_src $src_lang true`
	input_size=`python3 scripts/preprocess_translate.py $dev_infname_tgt $dev_outfname_tgt $tgt_lang true`
	echo "Number of sentences in dev: $input_size"
	# test preprocessing
	test_infname_src=$devtest_data_dir/en-${lang}/test.$src_lang
	test_infname_tgt=$devtest_data_dir/en-${lang}/test.$tgt_lang
	test_outfname_src=$devtest_norm_dir/test.$src_lang
	test_outfname_tgt=$devtest_norm_dir/test.$tgt_lang
	echo "Applying normalization and script conversion for test"
	input_size=`python3 scripts/preprocess_translate.py $test_infname_src $test_outfname_src $src_lang true`
	input_size=`python3 scripts/preprocess_translate.py $test_infname_tgt $test_outfname_tgt $tgt_lang true`
	echo "Number of sentences in test: $input_size"
done

# this concatenates lang pair data and creates text files to keep track of number of lines in each lang pair.
# this is imp as for joint training, we will merge all the lang pairs and the indivitual lang lines info
# would be required for adding specific lang tags later.

# the outputs of these scripts will  be text file like this:
# <lang1> <lang2> <number of lines>
# lang1-lang2 n1
# lang1-lang3 n2

echo $src_lang
echo $tgt_lang

python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'train'
python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'dev'
python3 scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list 'test'

if [[ "$reuse_bpe_vocab" == false ]]; then
	echo "Learning bpe. This will take a very long time depending on the size of the dataset"
	echo `date`
	if [[ "$vocab_type" == "sep" ]]; then
	    bash learn_single_bpe.sh $exp_dir
	else 
	    bash learn_bpe.sh $exp_dir
	fi
else
	echo "reusing old bpe"
	mkdir $exp_dir/final_bin
	# copy the old vocab
	cp -r $vocab_bpe_dir/vocab $exp_dir
	# copy the old dictionaries into the exp_dir/final_bin
	cp -r  $vocab_bpe_dir/final_bin/dict.* $exp_dir/final_bin
fi


echo `date`
echo "Applying bpe"

if [[ "$vocab_type" == "sep" ]]
then
    bash apply_single_bpe_traindevtest_notag.sh $exp_dir
else 
    bash apply_bpe_traindevtest_notag.sh $exp_dir
fi

mkdir -p $exp_dir/final
# # this is only required for joint training
# we apply language tags to the bpe segmented data
# if we are translating lang1 to lang2 then <lang1 line> will become __src__ <lang1> __tgt__ <lang2> <lang1 line>
echo "Adding language tags"
python3 scripts/add_joint_tags_translate.py $exp_dir 'train'
python3 scripts/add_joint_tags_translate.py $exp_dir 'dev'
python3 scripts/add_joint_tags_translate.py $exp_dir 'test'

# # this is important step if you are training with tpu and using num_batch_buckets
# # the currnet implementation does not remove outliers before bucketing and hence
# # removing these large sentences ourselves helps with getting better buckets
# python3 scripts/remove_large_sentences.py $exp_dir/bpe/train.SRC $exp_dir/bpe/train.TGT $exp_dir/final/train.SRC $exp_dir/final/train.TGT
# python3 scripts/remove_large_sentences.py $exp_dir/bpe/dev.SRC $exp_dir/bpe/dev.TGT $exp_dir/final/dev.SRC $exp_dir/final/dev.TGT
# python3 scripts/remove_large_sentences.py $exp_dir/bpe/test.SRC $exp_dir/bpe/test.TGT $exp_dir/final/test.SRC $exp_dir/final/test.TGT

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
		--thresholdsrc 5 \
		--memory-efficient-fp16
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
		--thresholdsrc 5 \
		--memory-efficient-fp16
fi
