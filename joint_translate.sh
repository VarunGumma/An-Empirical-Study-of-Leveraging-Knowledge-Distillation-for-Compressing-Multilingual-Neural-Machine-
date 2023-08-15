#!/bin/bash
echo `date`
infname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
exp_dir=$5
model=$6

### normalization and script conversion
echo -e "[INFO]\tApplying normalization and script conversion"
parallel --pipe --keep-order python scripts/preprocess_translate.py $src_lang true < $infname > $outfname.norm 

input_size=$(grep -c '.' $outfname.norm)
echo "Input size: ${input_size}"

### apply BPE to input file
echo -e "[INFO]\tApplying BPE"
parallel --pipe --keep-order subword-nmt apply-bpe \
    --codes $exp_dir/vocab/bpe_codes.32k.SRC \
    --vocabulary $exp_dir/vocab/vocab.SRC \
    --vocabulary-threshold 5 < $outfname.norm > $outfname._bpe

# not needed for joint training
echo -e "[INFO]\tAdding language tags"
parallel --pipe --keep-order bash scripts/add_tags_translate.sh $src_lang $tgt_lang < $outfname._bpe > $outfname.bpe

### run the model
echo -e "[INFO]\tDecoding"
fairseq-interactive $exp_dir/final_bin \
    --source-lang SRC \
    --target-lang TGT \
    --memory-efficient-fp16 \
    --path $exp_dir/$model/checkpoint_best.pt \
    --skip-invalid-size-inputs-valid-test \
    --batch-size 128 \
    --buffer-size 2500 \
    --beam 5 \
    --num-workers 24 \
    --remove-bpe \
    --input $outfname.bpe > $outfname.log 2>&1 

echo -e "[INFO]\tExtracting translations, script conversion and detokenization"
# this part reverses the transliteration from devnagiri script to target lang and then detokenizes it.
python scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang true 

echo -e "[INFO]\tPurging intermediate files"
rm $outfname.*

echo -e "[INFO]\tTranslation completed"