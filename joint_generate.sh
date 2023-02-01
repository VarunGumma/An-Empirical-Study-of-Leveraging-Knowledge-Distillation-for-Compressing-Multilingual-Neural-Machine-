#!/bin/bash
indirname=$1
outfname=$2
src_lang=$3
tgt_lang=$4
ckpt_path=$5
transliterate=${6:-true}

echo -e "[INFO]\tDecoding"

num_workers=`python3 -c "import multiprocessing; print(multiprocessing.cpu_count())"`

fairseq-generate \
$indirname/final_bin \
-s SRC -t TGT \
--distributed-world-size 1 \
--path $ckpt_path/checkpoint_best.pt \
--gen-subset train \
--batch-size 256 \
--beam 5 \
--max-len-a 1.2 \
--max-len-b 10 \
--remove-bpe \
--memory-efficient-fp16 \
--num-workers $num_workers \
--skip-invalid-size-inputs-valid-test > $outfname.log 2>&1

value=`cat $indirname/final_bin/preprocess.log | grep train.SRC`
array=($value)
input_size=${array[2]}

echo "Number of sentences: $input_size"
python3 scripts/postprocess_translate.py $outfname.log $outfname $input_size $tgt_lang $transliterate

# rm $outfname.*