#!/bin/bash
echo `date`

indirname=$1
outdirname=$2

rm -rf $outdirname

for lang in as bn gu hi kn ml mr or pa ta te; do
    echo "working on ${lang}"
    mkdir -p $outdirname/en-$lang
    cp $indirname/en-$lang/train.$lang $outdirname/en-$lang 
    bash joint_translate.sh $outdirname/en-$lang/train.$lang $outdirname/en-$lang/train.en $lang en checkpoints/it ../data_bin/v2_indic_en_bin
done