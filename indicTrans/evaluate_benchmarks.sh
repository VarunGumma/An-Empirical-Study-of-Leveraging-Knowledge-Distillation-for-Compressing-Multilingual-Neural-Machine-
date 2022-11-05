#!/bin/bash
echo `date`

allArgs=("$@")

for ext in "${allArgs[@]}"; do
    # ufal
    echo ">>>>> UFAL"
    ./joint_translate.sh $ext ../benchmarks/ufal-ta/en-ta/test.ta ../benchmarks/ufal-ta/en-ta/outfile.en ta en ../checkpoints ../indic-en-exp
    output=$(./compute_bleu.sh ../benchmarks/ufal-ta/en-ta/outfile.en ../benchmarks/ufal-ta/en-ta/test.en ta en)
    echo -e "ufal - ta: ${output}\n" > ../results/$ext.txt
    echo -e "<<<<< UFAL\n"

    # PMI
    echo ">>>>> PMI"
    ./joint_translate.sh $ext ../benchmarks/pmi/en-as/test.as ../benchmarks/pmi/en-as/outfile.en as en ../checkpoints ../indic-en-exp
    output=$(./compute_bleu.sh ../benchmarks/pmi/en-as/outfile.en ../benchmarks/pmi/en-as/test.en as en)
    echo -e "pmi - as: ${output}\n" >> ../results/$ext.txt
    echo -e "<<<<< PMI\n"

    # wat2021
    echo ">>>>> WAT2021"
    for lang in bn gu hi kn ml mr or pa ta te; do
        ./joint_translate.sh $ext ../benchmarks/wat2021-devtest/en-$lang/test.$lang ../benchmarks/wat2021-devtest/en-$lang/outfile.en $lang en ../checkpoints ../indic-en-exp 
        output=$(./compute_bleu.sh ../benchmarks/wat2021-devtest/en-$lang/outfile.en ../benchmarks/wat2021-devtest/en-$lang/test.en $lang en)
        echo -e "wat2021 - ${lang}: ${output}\n" >> ../results/$ext.txt
    done
    echo -e  "<<<<< WAT2021\n"

    # wat2020
    echo ">>>>> WAT2020"
    for lang in bn gu hi ml mr ta te; do
        ./joint_translate.sh $ext ../benchmarks/wat2020-devtest/en-$lang/test.$lang ../benchmarks/wat2020-devtest/en-$lang/outfile.en $lang en ../checkpoints ../indic-en-exp
        output=$(./compute_bleu.sh ../benchmarks/wat2020-devtest/en-$lang/outfile.en ../benchmarks/wat2020-devtest/en-$lang/test.en $lang en)
        echo -e "wat2020 - ${lang}: ${output}\n" >> ../results/$ext.txt
    done 
    echo -e "<<<<< WAT2020\n"

    # wmt-news
    echo ">>>>> WMT-NEWS"
    for lang in gu hi ta; do
        ./joint_translate.sh $ext ../benchmarks/wmt-news/en-$lang/test.$lang ../benchmarks/wmt-news/en-$lang/outfile.en $lang en ../checkpoints ../indic-en-exp
        output=$(./compute_bleu.sh ../benchmarks/wmt-news/en-$lang/outfile.en ../benchmarks/wmt-news/en-$lang/test.en $lang en)
        echo -e "wmt-news - ${lang}: ${output}\n" >> ../results/$ext.txt
    done 
    echo -e "<<<<< WMT-NEWS\n"

    # flores101
    echo ">>>>> FLORES-101"
    for lang in as bn gu hi kn ml mr or pa ta te; do
        ./joint_translate.sh $ext ../benchmarks/flores101_dataset/en-$lang/test.$lang ../benchmarks/flores101_dataset/en-$lang/outfile.en $lang en ../checkpoints ../indic-en-exp
        output=$(./compute_bleu.sh ../benchmarks/flores101_dataset/en-$lang/outfile.en ../benchmarks/flores101_dataset/en-$lang/test.en $lang en)
        echo -e "flores101_dataset - ${lang}: ${output}\n" >> ../results/$ext.txt
    done
    echo -e "<<<<< FLORES-101\n"

    echo -e "[INFO]\t${ext} evaluation completed\n\n"
done

echo -e "[INFO]\tconverting all txt files to csv"
python3 scripts/convert_txt_to_csv.py "$@"
