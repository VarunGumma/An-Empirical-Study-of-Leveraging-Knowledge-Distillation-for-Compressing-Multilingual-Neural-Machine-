#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
model=$6
transliterate=$7

SRC_PREFIX='SRC'
TGT_PREFIX='TGT'
SUBWORD_NMT_DIR='subword-nmt'

echo $exp_dir

### normalization and script conversion

echo -e "[INFO]\tApplying normalization and script conversion"
input_size=`python scripts/preprocess_translate.py $infname $outfname.norm $src_lang $transliterate`
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

# declare -A lang_familes=( ["as"]="as+bn+or" ["bn"]="as+bn+or" ["gu"]="gu+hi+mr+pa" ["hi"]="gu+hi+mr+pa" ["kn"]="kn+ml+ta+te" ["ml"]="kn+ml+ta+te" ["mr"]="gu+hi+mr+pa" ["or"]="as+bn+or" ["pa"]="gu+hi+mr+pa" ["ta"]="kn+ml+ta+te" ["te"]="kn+ml+ta+te" )

CUDA_VISIBLE_DEVICES=7 fairseq-interactive $exp_dir/final_bin \
--source-lang $SRC_PREFIX \
--target-lang $TGT_PREFIX \
--path $exp_dir/$model/checkpoint_best.pt \
--batch-size 128 \
--buffer-size 2500 \
--beam 5 \
--max-len-a 1.2 \
--max-len-b 10 \
--remove-bpe \
--skip-invalid-size-inputs-valid-test \
--input $outfname.bpe \
--num-workers 16 \
--user-dir model_configs \
--memory-efficient-fp16  >  $outfname.log 2>&1

echo -e "[INFO]\tExtracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python3 scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang $transliterate

echo -e "[INFO]\tTranslation completed"