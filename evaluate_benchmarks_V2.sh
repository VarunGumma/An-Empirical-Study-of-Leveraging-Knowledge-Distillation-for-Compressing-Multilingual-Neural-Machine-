echo ">>>>> FLORES-101"
for lang in as bn gu hi kn ml mr or pa ta te; do
    bash joint_translate_xx.sh \
    ~/benchmarks/flores101_dataset/en-$lang/dev.$lang \
    ~/benchmarks/flores101_dataset/en-$lang/outfile.en $lang en \
    ~/indicTrans_ckpts/indic-en/model \
    ~/indicTrans_ckpts/indic-en \
    ~/benchmarks/flores101_dataset/en-$lang/encoder_states
done
echo -e "<<<<< FLORES-101\n"

echo ">>>>> WAT2021"
for lang in bn gu hi kn ml mr or pa ta te; do
    bash joint_translate_xx.sh \
    ~/benchmarks/wat2021-devtest/en-$lang/dev.$lang \
    ~/benchmarks/wat2021-devtest/en-$lang/outfile.en $lang en \
    ~/indicTrans_ckpts/indic-en/model \
    ~/indicTrans_ckpts/indic-en \
    ~/benchmarks/wat2021-devtest/en-$lang/encoder_states
done
echo -e  "<<<<< WAT2021\n"

echo ">>>>> WAT2020"
for lang in bn gu hi ml mr ta te; do
    bash joint_translate_xx.sh \
    ~/benchmarks/wat2020-devtest/en-$lang/dev.$lang \
    ~/benchmarks/wat2020-devtest/en-$lang/outfile.en $lang en \
    ~/indicTrans_ckpts/indic-en/model \
    ~/indicTrans_ckpts/indic-en \
    ~/benchmarks/wat2020-devtest/en-$lang/encoder_states
done 
echo -e "<<<<< WAT2020\n"