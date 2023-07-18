#!/bin/bash

expdir=$1  # EXPDIR
num_operations=${2:-32000}

data_dir="$expdir/data"
train_file="$data_dir/train"

echo "Input file: ${train_file}"

mkdir -p $expdir/vocab

echo "learning joint BPE"
cat $train_file.SRC  $train_file.TGT > $train_file.ALL
subword-nmt learn_bpe -i $train_file.ALL -s $num_operations -o $expdir/vocab/bpe_codes.32k.SRC_TGT --num-workers -1

echo "computing SRC vocab"
subword-nmt apply-bpe -c $expdir/vocab/bpe_codes.32k.SRC_TGT -i $train_file.SRC --num-workers -1 | subword-nmt get-vocab > $expdir/vocab/vocab.tmp.SRC
parallel --pipe --keep-order python scripts/clean_vocab.py < $expdir/vocab/vocab.tmp.SRC > $expdir/vocab/vocab.SRC

echo "computing TGT vocab"
subword-nmt apply-bpe -c $expdir/vocab/bpe_codes.32k.SRC_TGT -i $train_file.TGT --num-workers -1 | subword-nmt get-vocab > $expdir/vocab/vocab.tmp.TGT
parallel --pipe --keep-order python scripts/clean_vocab.py < $expdir/vocab/vocab.tmp.TGT > $expdir/vocab/vocab.TGT

rm $train_file.ALL $expdir/vocab/vocab.tmp.TGT $expdir/vocab/vocab.tmp.SRC