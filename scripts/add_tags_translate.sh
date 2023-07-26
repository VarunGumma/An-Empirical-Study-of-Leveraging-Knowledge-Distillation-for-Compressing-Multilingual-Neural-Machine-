#!/bin/bash

# Assign command line arguments to variables
src_lang=$1
tgt_lang=$2

while read -r line; do
    # Add tokens and write to output file
    echo "__src__${src_lang}__ __tgt__${tgt_lang}__ ${line}"
done 