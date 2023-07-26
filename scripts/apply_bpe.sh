#!/bin/bash

expdir=$1  # EXPDIR
dset=$2 # split to apply bpe and vocab

data_dir="$expdir/data"
mkdir -p $expdir/bpe

echo $dset
in_dset_dir="$data_dir/$dset"
out_dset_dir="$expdir/bpe/$dset"

echo -e "\t- Apply joint vocab to SRC corpus"
parallel --pipe --keep-order subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.SRC_TGT \
    --vocabulary $expdir/vocab/vocab.SRC \
    --vocabulary-threshold 5 < $in_dset_dir.SRC > $out_dset_dir.SRC

echo -e "\t- Apply joint vocab to TGT corpus"
parallel --pipe --keep-order subword-nmt apply-bpe \
    --codes $expdir/vocab/bpe_codes.32k.SRC_TGT \
    --vocabulary $expdir/vocab/vocab.TGT \
    --vocabulary-threshold 5 < $in_dset_dir.TGT > $out_dset_dir.TGT
