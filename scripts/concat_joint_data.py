import os
from tqdm import tqdm
import sys

INDIC_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]


def concat_data(data_dir, outdir, lang_pair_list,
                out_src_lang='SRC', out_trg_lang='TGT', split='train'):
    """
    data_dir: input dir, contains directories for language pairs named l1-l2
    """
    os.makedirs(outdir, exist_ok=True)

    out_src_fname = f'{outdir}/{split}.{out_src_lang}'
    out_trg_fname = f'{outdir}/{split}.{out_trg_lang}'

    print('\n', out_src_fname)
    print(out_trg_fname)

    # concatenate train data
    if os.path.isfile(out_src_fname):
        os.unlink(out_src_fname)
    if os.path.isfile(out_trg_fname):
        os.unlink(out_trg_fname)

    for src_lang, trg_lang in (pbar := tqdm(lang_pair_list)):
        in_src_fname = f'{data_dir}/{src_lang}-{trg_lang}/{split}.{src_lang}'
        in_trg_fname = f'{data_dir}/{src_lang}-{trg_lang}/{split}.{trg_lang}'
        
        if not os.path.exists(in_src_fname):
            continue
        if not os.path.exists(in_trg_fname): 
            continue

        os.system(f'cat {in_src_fname} >> {out_src_fname}')
        os.system(f'cat {in_trg_fname} >> {out_trg_fname}')
        
        pbar.set_description(
            f'src: {src_lang}, tgt: {tgt_lang}, in_src_fname: {in_src_fname}, in_trg_fname: {in_trg_fname}'
        )

    corpus_stats(data_dir, outdir, lang_pair_list, split)


def corpus_stats(data_dir, outdir, lang_pair_list, split):
    """
    data_dir: input dir, contains directories for language pairs named l1-l2
    """

    with open(f'{outdir}/{split}_lang_pairs.txt', 'w', encoding='utf-8') as lpfile:

        for src_lang, trg_lang in (pbar := tqdm(lang_pair_list)):
            print(f'src: {src_lang}, tgt:{trg_lang}')

            in_src_fname = f'{data_dir}/{src_lang}-{trg_lang}/{split}.{src_lang}'
            
            if not os.path.exists(in_src_fname):
                continue

            with open(in_src_fname, 'r', encoding='utf-8') as infile:
                corpus_size = sum(map(lambda x: 1, infile))

            lpfile.write(f'{src_lang}\t{trg_lang}\t{corpus_size}\n')
            pbar.set_description(in_src_fname)


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]
    languages_list = sys.argv[5]
    split = sys.argv[6]
    
    INDIC_LANGS = languages_list.split('+')
    lang_pair_list = [(['en', lang] if src_lang == 'en' else [lang, 'en']) for lang in INDIC_LANGS]

    concat_data(in_dir, out_dir, lang_pair_list, split=split)

