import torch
from os import listdir
from sys import argv
from engine import Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'
devtest_dir, exp_dir, ckpt_dir, src, tgt, beam_size, batch_size, model_list = argv[1:]

for model_name in model_list.split(','):
    print(f"[INFO]\tevaluating {model_name}")
    model = Model(exp_dir=exp_dir, ckpt_dir=f"{ckpt_dir}/{model_name}", device=device)

    for lang_pair in sorted(listdir(devtest_dir)):
        if src == 'en':
            tgt = lang_pair.split('-')[1]
        else:
            src = lang_pair.split('-')[1]

        with open(f"{devtest_dir}/{lang_pair}/test.{src}", 'r', encoding='utf-8') as f:
            src_sents = [line.strip() for line in f]

        translated_sents = model.batch_translate(
            src_sents,
            src_lang=src,
            tgt_lang=tgt,
            beam_size=int(beam_size),
            max_batch_size=int(batch_size)
        )

        with open(f"{devtest_dir}/{lang_pair}/outfile.{tgt}", 'w', encoding='utf-8') as f:
            f.write('\n'.join(translated_sents))
    
    system(f"bash ../compute_bleu.sh {devtest_dir}/{lang_pair}/test.{src} {devtest_dir}/{lang_pair}/test.{src} {src} {tgt} > results/{lang_pair}.json"")