#!/bin/bash

expdir=$1  # EXPDIR
dset=$2 # split to apply bpe and vocab

SUBWORD_NMT_DIR="subword-nmt"

mkdir -p $expdir/bpe
in_dset_dir="$expdir/data/$dset"
out_dset_dir="$expdir/bpe/$dset"

echo -e "\t- Apply vocab to SRC corpus"
parallel --pipe --keep-order subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.SRC \
    --vocabulary $expdir/vocab/vocab.SRC \
    --vocabulary-threshold 5 < $in_dset_dir.SRC > $out_dset_dir.SRC


echo -e "\t- Apply vocab to TGT corpus"
parallel --pipe --keep-order subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.TGT \
    --vocabulary $expdir/vocab/vocab.TGT \
    --vocabulary-threshold 5 < $in_dset_dir.TGT > $out_dset_dir.TGT

