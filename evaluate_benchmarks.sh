#!/bin/bash
echo `date`

devtest_dir=$1
ckpt_base_dir=$2
exp_dir=$3; shift

allArgs=("$@")

for ext in "${allArgs[@]}"; do
    # ufal
    echo ">>>>> UFAL"
    bash joint_translate.sh $devtest_dir/ufal-ta/en-ta/test.ta $devtest_dir/ufal-ta/en-ta/outfile.en ta en $ckpt_base_dir/$ext $exp_dir
    output=$(bash compute_bleu.sh $devtest_dir/ufal-ta/en-ta/outfile.en $devtest_dir/ufal-ta/en-ta/test.en ta en)
    echo -e "ufal - ta: ${output}\n" > results/$ext.txt
    echo -e "<<<<< UFAL\n"

    # PMI
    echo ">>>>> PMI"
    bash joint_translate.sh $devtest_dir/pmi/en-as/test.as $devtest_dir/pmi/en-as/outfile.en as en $ckpt_base_dir/$ext $exp_dir
    output=$(bash compute_bleu.sh $devtest_dir/pmi/en-as/outfile.en $devtest_dir/pmi/en-as/test.en as en)
    echo -e "pmi - as: ${output}\n" >> results/$ext.txt
    echo -e "<<<<< PMI\n"

    # wat2021
    echo ">>>>> WAT2021"
    for lang in bn gu hi kn ml mr or pa ta te; do
        bash joint_translate.sh $devtest_dir/wat2021-devtest/en-$lang/test.$lang $devtest_dir/wat2021-devtest/en-$lang/outfile.en $lang en $ckpt_base_dir/$ext $exp_dir 
        output=$(bash compute_bleu.sh $devtest_dir/wat2021-devtest/en-$lang/outfile.en $devtest_dir/wat2021-devtest/en-$lang/test.en $lang en)
        echo -e "wat2021 - ${lang}: ${output}\n" >> results/$ext.txt
    done
    echo -e  "<<<<< WAT2021\n"

    # wat2020
    echo ">>>>> WAT2020"
    for lang in bn gu hi ml mr ta te; do
        bash joint_translate.sh $devtest_dir/wat2020-devtest/en-$lang/test.$lang $devtest_dir/wat2020-devtest/en-$lang/outfile.en $lang en $ckpt_base_dir/$ext $exp_dir
        output=$(bash compute_bleu.sh $devtest_dir/wat2020-devtest/en-$lang/outfile.en $devtest_dir/wat2020-devtest/en-$lang/test.en $lang en)
        echo -e "wat2020 - ${lang}: ${output}\n" >> results/$ext.txt
    done 
    echo -e "<<<<< WAT2020\n"

    # wmt-news
    echo ">>>>> WMT-NEWS"
    for lang in gu hi ta; do
        bash joint_translate.sh $devtest_dir/wmt-news/en-$lang/test.$lang $devtest_dir/wmt-news/en-$lang/outfile.en $lang en $ckpt_base_dir/$ext $exp_dir
        output=$(bash compute_bleu.sh $devtest_dir/wmt-news/en-$lang/outfile.en $devtest_dir/wmt-news/en-$lang/test.en $lang en)
        echo -e "wmt-news - ${lang}: ${output}\n" >> results/$ext.txt
    done 
    echo -e "<<<<< WMT-NEWS\n"

    # flores101
    echo ">>>>> FLORES-101"
    for lang in as bn gu hi kn ml mr or pa ta te; do
        bash joint_translate.sh $devtest_dir/flores101_dataset/en-$lang/test.$lang $devtest_dir/flores101_dataset/en-$lang/outfile.en $lang en $ckpt_base_dir/$ext $exp_dir
        output=$(bash compute_bleu.sh $devtest_dir/flores101_dataset/en-$lang/outfile.en $devtest_dir/flores101_dataset/en-$lang/test.en $lang en)
        echo -e "flores101_dataset - ${lang}: ${output}\n" >> results/$ext.txt
    done
    echo -e "<<<<< FLORES-101\n"

    echo -e "[INFO]\t${ext} evaluation completed\n\n"
done

echo -e "[INFO]\tconverting all txt files to csv"
python3 convert_txt_to_csv.py "$@"
