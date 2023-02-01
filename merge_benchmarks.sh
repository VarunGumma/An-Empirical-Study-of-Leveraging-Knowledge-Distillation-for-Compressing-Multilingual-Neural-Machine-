#!/bin/bash
echo `date`
base_path=$1
benchmarks=$2
languages=$3

rm -rf $base_path/all
mkdir $base_path/all

# languages separated by '+'
IFS='+' read -ra languages_split <<< $languages
IFS='+' read -ra benchmarks_split <<< $benchmarks

for lang in "${languages_split[@]}"; do
    mkdir $base_path/all/en-$lang
    for devtest_dir in "${benchmarks_split[@]}"; do
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