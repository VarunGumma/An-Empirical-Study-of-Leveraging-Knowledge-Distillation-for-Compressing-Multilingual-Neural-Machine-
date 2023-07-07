#!/bin/bash

# Assign command line arguments to variables
src_lang=$1
tgt_lang=$2

while read -r line; do
    # Add tokens and write to output file
    echo "__src__${src_lang} __tgt__${tgt_lang} ${line}"
done 