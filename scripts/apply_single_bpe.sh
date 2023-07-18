#!/bin/bash

expdir=$1  # EXPDIR
dset=$2 # split to apply bpe and vocab

SUBWORD_NMT_DIR="subword-nmt"

data_dir="$expdir/data"
mkdir -p $expdir/bpe

echo $dset
in_dset_dir="$data_dir/$dset"
out_dset_dir="$expdir/bpe/$dset"

echo "Apply to SRC corpus"
subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.SRC \
    --vocabulary $expdir/vocab/vocab.SRC \
    --vocabulary-threshold 5 \
    --num-workers -1 < $in_dset_dir.SRC > $out_dset_dir.SRC


echo "Apply to TGT corpus"
subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.TGT \
    --vocabulary $expdir/vocab/vocab.TGT \
    --vocabulary-threshold 5 \
    --num-workers -1 < $in_dset_dir.TGT > $out_dset_dir.TGT

