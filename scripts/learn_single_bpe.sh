#!/bin/bash

expdir=$1  # EXPDIR
num_operations=${2:-32000}

data_dir="$expdir/data"
train_file="$data_dir/train"

echo "Input file: ${train_file}"

mkdir -p $expdir/vocab

echo "learning source BPE"
subword-nmt learn-bpe -i $train_file.SRC -s $num_operations -o $expdir/vocab/bpe_codes.32k.SRC --num-workers -1

echo "learning target BPE"
subword-nmt learn-bpe -i $train_file.TGT -s $num_operations -o $expdir/vocab/bpe_codes.32k.TGT --num-workers -1

echo "computing SRC vocab"
subword-nmt apply-bpe -c $expdir/vocab/bpe_codes.32k.SRC -i $train_file.SRC --num-workers -1 | subword-nmt get-vocab > $expdir/vocab/vocab.tmp.SRC
parallel --pipe --keep-order python scripts/clean_vocab.py < $expdir/vocab/vocab.tmp.SRC > $expdir/vocab/vocab.SRC

echo "computing TGT vocab"
subword-nmt apply-bpe -c $expdir/vocab/bpe_codes.32k.TGT -i $train_file.TGT --num-workers -1 | subword-nmt get-vocab > $expdir/vocab/vocab.tmp.TGT
parallel --pipe --keep-order python scripts/clean_vocab.py < $expdir/vocab/vocab.tmp.TGT > $expdir/vocab/vocab.TGT

rm $expdir/vocab/vocab.tmp.TGT $expdir/vocab/vocab.tmp.SRC