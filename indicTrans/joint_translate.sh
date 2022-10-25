#!/bin/bash
echo `date`
ext=$1
infname=$2
outfname=$3
src_lang=$4
tgt_lang=$5
exp_dir=$6
ref_fname=$7

SRC_PREFIX='SRC'
TGT_PREFIX='TGT'

SUBWORD_NMT_DIR='subword-nmt'
data_bin_dir=$exp_dir/final_bin
model_dir=$exp_dir/model-$ext

### normalization and script conversion

echo -e "[INFO]\tApplying normalization and script conversion"
input_size=`python scripts/preprocess_translate.py $infname $outfname.norm $src_lang true`
echo -e "[INFO]\tNumber of sentences in input: $input_size"

### apply BPE to input file

echo -e "[INFO]\tApplying BPE"
python3 $SUBWORD_NMT_DIR/subword_nmt/apply_bpe.py \
    -c $exp_dir/vocab/bpe_codes.32k.$SRC_PREFIX \
    --vocabulary $exp_dir/vocab/vocab.$SRC_PREFIX \
    --vocabulary-threshold 5 < $outfname.norm > $outfname._bpe

# not needed for joint training
# echo "Adding language tags"
python3 scripts/add_tags_translate.py $outfname._bpe $outfname.bpe $src_lang $tgt_lang

### run decoder

echo -e "[INFO]\tDecoding"

src_input_bpe_fname=$outfname.bpe
tgt_output_fname=$outfname

fairseq-interactive $data_bin_dir \
    -s $SRC_PREFIX -t $TGT_PREFIX \
    --distributed-world-size 2 \
    --path $model_dir/checkpoint_best.pt \
    --batch-size 512 --buffer-size 2500 --beam 5 --remove-bpe \
    --skip-invalid-size-inputs-valid-test \
    --input $src_input_bpe_fname \
    --num-workers 64 \
    --memory-efficient-fp16  >  $tgt_output_fname.log 2>&1


echo -e "[INFO]\tExtracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python3 scripts/postprocess_translate.py $tgt_output_fname.log $tgt_output_fname $input_size $tgt_lang true

echo -e "[INFO]\tTranslation completed"
