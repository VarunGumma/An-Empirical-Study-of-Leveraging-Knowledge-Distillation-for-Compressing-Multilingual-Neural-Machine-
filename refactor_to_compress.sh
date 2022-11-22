#!/bin/bash

echo `date`
src_dir=$1
dest_dir=$2

rm -rf $dest_dir
mkdir $dest_dir

for lang in as bn gu hi kn ml mr or pa ta te; do 
    mkdir $dest_dir/en-$lang
    cp $src_dir/v2_100/train.$lang $dest_dir/en-$lang/train.$lang
    for idx in 0 10 20 30 40 50 60 70 80 90 100; do
        cp $src_dir/v2_$idx/train.en $dest_dir/en-$lang/train.en_$idx
    done
done