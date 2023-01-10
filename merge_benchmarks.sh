#!/bin/bash
echo `date`
base_path=$1
languages=$2
shift 2

# list of all benchmarks to merge and form validation set
allArgs=("$@")

rm -rf $base_path/all
mkdir $base_path/all

# languages separated by '+'
IFS='+' read -ra languages_split <<< $languages

for lang in "${languages_split[@]}"; do
    mkdir $base_path/all/en-$lang
    for devtest_dir in "${allArgs[@]}"; do
        if [[ -f $base_path/$devtest_dir/en-$lang/dev.$lang ]]; then
            cat $base_path/$devtest_dir/en-$lang/dev.$lang >> $base_path/all/en-$lang/dev.$lang
            cat $base_path/$devtest_dir/en-$lang/dev.en >> $base_path/all/en-$lang/dev.en
        fi
        if [[ -f $base_path/$devtest_dir/en-$lang/test.$lang ]]; then
            cat $base_path/$devtest_dir/en-$lang/test.$lang >> $base_path/all/en-$lang/test.$lang
            cat $base_path/$devtest_dir/en-$lang/test.en >> $base_path/all/en-$lang/test.en
        fi
    done
done