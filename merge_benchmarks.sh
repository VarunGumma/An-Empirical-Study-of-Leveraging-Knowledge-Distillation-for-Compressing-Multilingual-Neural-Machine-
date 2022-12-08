#!/bin/bash
echo `date`
base_path=$1
shift 1

# list of all benchmarks to merge and form validation set
allArgs=("$@")

rm -rf $base_path/all
mkdir $base_path/all

for lang in as bn gu hi kn ml mr or pa ta te; do
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