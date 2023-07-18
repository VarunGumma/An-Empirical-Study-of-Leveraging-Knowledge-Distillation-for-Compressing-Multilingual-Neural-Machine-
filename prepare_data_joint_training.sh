#!/bin/bash

exp_dir=$1
train_dir=$2
devtest_dir=$3
vocab_dir=$4
src_lang=$5
tgt_lang=$6
languages_list=$7

echo `date`
# remove old copies
rm -rf $exp_dir
rm -rf ${train_dir}_copy

echo -e "Removing overlap between train and devtest"
echo -e "This will create a temporary copy of the train directory, but will remove it at the end of the program"

cp -r $train_dir ${train_dir}_copy
train_dir=${train_dir}_copy

python scripts/remove_train_devtest_overlaps.py -t $train_dir -d $devtest_dir -l $languages_list

mkdir -p $exp_dir/data
mkdir -p $exp_dir/final
mkdir -p $exp_dir/final_bin

# languages separated by '+'
IFS='+' read -ra langs <<< $languages_list

for lang in ${langs[@]}; do
    # copy only data you want to work with
    # saves space and is more efficient
	echo -e "Copying en-${lang} data"
    cp -r $train_dir/en-$lang $exp_dir
	cp -r $devtest_dir/en-$lang/dev.* $exp_dir/en-$lang
done

# copy the old vocab
echo -e "Copying vocab and fairseq dictionaries"
cp -r $vocab_dir/vocab $exp_dir
cp -r $vocab_dir/final_bin/dict.* $exp_dir/final_bin


echo -e "Preparing data"

echo "exp_dir: ${exp_dir}, ${src_lang}-${tgt_lang}"

IFS='+' read -ra langs <<< $languages_list

for lang in ${langs[@]}; do
	if [[ "$src_lang" == en ]]; then
		tgt_lang=$lang
	else
		src_lang=$lang
	fi

	echo "Working on '$src_lang'"

	norm_dir=$exp_dir/norm/$src_lang-$tgt_lang
	mkdir -p $norm_dir

	# train preprocessing
	train_infname_src=$exp_dir/en-${lang}/train.$src_lang
	train_infname_tgt=$exp_dir/en-${lang}/train.$tgt_lang
	train_outfname_src=$norm_dir/train.$src_lang
	train_outfname_tgt=$norm_dir/train.$tgt_lang
	echo "Applying normalization and script conversion for train"
	parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang true < $train_infname_src > $train_outfname_src 
	parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang true < $train_infname_tgt > $train_outfname_tgt

	# dev preprocessing
	dev_infname_src=$exp_dir/en-${lang}/dev.$src_lang
	dev_infname_tgt=$exp_dir/en-${lang}/dev.$tgt_lang
	dev_outfname_src=$norm_dir/dev.$src_lang
	dev_outfname_tgt=$norm_dir/dev.$tgt_lang
	echo "Applying normalization and script conversion for dev"
	parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang true < $dev_infname_src > $dev_outfname_src 
	parallel --pipe --keep-order python scripts/preprocess_translate.py $tgt_lang true < $dev_infname_tgt > $dev_outfname_tgt 
done


for split in train dev; do
	# this concatenates lang pair data and creates text files to keep track of number of lines in each lang pair.
	# this is imp as for joint training, we will merge all the lang pairs and the indivitual lang lines info
	# would be required for adding specific lang tags later.

	# the outputs of these scripts will  be text file like this:
	# <lang1> <lang2> <number of lines>
	# lang1-lang2 n1
	# lang1-lang3 n2
	echo "Merging ${split} data of all languages"
	python scripts/concat_joint_data.py $exp_dir/norm $exp_dir/data $src_lang $tgt_lang $languages_list $split

	# Apply BPE to concatenated data
	echo "Applying bpe to ${split}"
	bash scripts/apply_single_bpe.sh $exp_dir $split

	# # this is only required for joint training
	# we apply language tags to the bpe segmented data
	# if we are translating lang1 to lang2 then <lang1 line> will become __src__ <lang1> __tgt__ <lang2> <lang1 line>
	echo "Adding language tags to ${split}"
	python scripts/add_joint_tags_translate.py $exp_dir $split
done

# Binarize the training data for using with fairseq train
bash scripts/fairseq_binarize.sh $exp_dir/final $exp_dir/final_bin

echo -e "cleaning unnecessary files from exp dir to save space"
rm -rf $exp_dir/bpe $exp_dir/devtest $exp_dir/final $exp_dir/data $exp_dir/norm $exp_dir/en-* $train_dir