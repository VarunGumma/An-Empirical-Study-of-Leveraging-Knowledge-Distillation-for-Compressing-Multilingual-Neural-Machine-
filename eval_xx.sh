echo ">>>>> FLORES-101"
for lang in as bn gu hi kn ml mr or pa ta te; do
    bash joint_translate_xx.sh \
    ../flores101_dataset/en-$lang/dev.$lang \
    ../flores101_dataset/en-$lang/outfile.en $lang en \
    ../indic-en/model \
    ../indic-en \
    ~/Downloads/flores101_dataset/en-$lang/enc_states
done
echo -e "<<<<< FLORES-101\n"