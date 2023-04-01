devtest_data_dir=$1
indic2en_exp_dir=$2
indic2en_ckpt_dir=$3
en2indic_exp_dir=$4
en2indic_ckpt_dir=$5

metrics_dir="pivoting-bleu-metrics"

mkdir -p $metrics_dir

langs=(as bn hi gu kn ml mr or pa ta te)

for src_lang in ${langs[@]}; do
    for tgt_lang in ${langs[@]}; do
        if [ $src_lang == $tgt_lang ]; then
            continue
        fi

        echo "$src_lang - $tgt_lang"

        # indic to en
        echo "Generating Translations from ${src_lang} to en"
        bash joint_translate.sh $devtest_data_dir/$src_lang-$tgt_lang/test.$src_lang $devtest_data_dir/$src_lang-$tgt_lang/test.en.pred.itv2 $src_lang en $indic2en_ckpt_dir $indic2en_exp_dir true

        # en to indic
        echo "Generating Translations from en to ${tgt_lang}"
        bash joint_translate.sh $devtest_data_dir/$src_lang-$tgt_lang/test.en.pred.itv2 $devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang.pred.itv2 en $tgt_lang $en2indic_ckpt_dir $en2indic_exp_dir true

        echo "Computing Metrics"
        bash compute_bleu.sh $devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang.pred.itv2 $devtest_data_dir/$src_lang-$tgt_lang/test.$tgt_lang $tgt_lang $tgt_lang > $metrics_dir/$src_lang-$tgt_lang.txt
    done
done
