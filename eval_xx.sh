echo ">>>>> FLORES-101"
for lang in as bn gu hi kn ml mr or pa ta te; do
    bash joint_translate_xx.sh \
    ~/MTP/flores101_dataset/en-$lang/dev.$lang \
    ~/MTP/flores101_dataset/en-$lang/outfile.en $lang en \
    ~/MTP/indic-en/model \
    ~/MTP/indic-en \
    ~/MTP/flores101_dataset/en-$lang/encoder_states
done
echo -e "<<<<< FLORES-101\n"