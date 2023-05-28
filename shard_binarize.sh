#!/bin/bash

echo `date`
exp_dir=$1
vocab_dir=$2
src_lang=$3
tgt_lang=$4
num_shards=$5
train_data_dir=$6

echo "Running data preparation ${exp_dir}"

if [ "$src_lang" = "en" ]; then
    lang="$tgt_lang"
else
    lang="$src_lang"
fi

pair=en-$lang
echo $pair

SUBWORD_NMT_DIR='subword-nmt'

infname=$train_data_dir/$pair/train.$src_lang

input_size=$(awk 'END{print NR}' $infname)
echo "Number of sentences: $input_size"

window_size=$((input_size / num_shards))
echo "Number of sentences in each shard: $window_size"

num_examples=1
shard_id=0

while [[ $num_examples -le $input_size ]]; do
    start=$num_examples
    num_examples=$((num_examples + window_size))
    end="$((num_examples > input_size ? input_size : num_examples))"
    echo "$start - $end"

    shard_infname=$infname.p${shard_id}
    out_data_dir=$exp_dir/binarized_train_data_only/${src_lang}_${tgt_lang}_p${shard_id}_final
    
    sed -n -e "$start,$end p" -e "$end q" $infname > $shard_infname

    echo "Applying normalization and script conversion"
    python3 scripts/preprocess_translate.py $shard_infname $shard_infname.norm $src_lang true

    echo "Applying BPE ..."
    parallel --pipe --keep-order \
        python3 $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
        -c $exp_dir/vocab/bpe_codes.32k.SRC \
        --vocabulary $exp_dir/vocab/vocab.TGT \
        --vocabulary-threshold 5 \
        --num-workers "-1" \
        < $shard_infname.norm
        > $shard_infname._bpe

    python3 scripts/add_tags_translate.py $shard_infname._bpe $shard_infname.bpe $src_lang $tgt_lang

    mkdir -p $out_data_dir

    src_infname=$shard_infname.bpe
    src_outfname=$out_data_dir/train.SRC
    tgt_outfname=$out_data_dir/train.TGT
    outfname=$out_data_dir/train.$src_lang.p${shard_id}

    # Copy the processed files for binarization
    cp -r $src_infname $src_outfname
    cp -r $src_infname $tgt_outfname
    cp -r $shard_infname $outfname

    num_examples=$((num_examples + 1))
    shard_id=$((shard_id + 1))

    echo "Binarizing data ..."
    # Binarize the training data for using with fairseq train

    # use cpu_count to get num_workers instead of setting it manually when running in different instances
    num_workers=`python -c "import multiprocessing; print(multiprocessing.cpu_count())"`

    fairseq-preprocess \
        --source-lang SRC \
        --target-lang TGT \
        --trainpref $out_data_dir/train \
        --destdir ${out_data_dir}_bin \
        --workers $num_workers \
        --srcdict $exp_dir/final_bin/dict.SRC.txt \
        --tgtdict $exp_dir/final_bin/dict.TGT.txt \
        --thresholdtgt 5 \
		--thresholdsrc 5

    rm -rf $out_data_dir

done