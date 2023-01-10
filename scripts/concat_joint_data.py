import os
from tqdm import tqdm
import sys

INDIC_LANGS = ["as", "bn", "gu", "hi", "kn", "ml", "mr", "or", "pa", "ta", "te"]

def add_token(sent, tag_infos):
    """ add special tokens specified by tag_infos to each element in list

    tag_infos: list of tuples (tag_type,tag)

    each tag_info results in a token of the form: __{tag_type}__{tag}__

    """

    tokens = []
    for tag_type, tag in tag_infos:
        token = '__' + tag_type + '__' + tag + '__'
        tokens.append(token)

    return ' '.join(tokens) + ' ' + sent


def concat_data(data_dir, outdir, lang_pair_list,
                out_src_lang='SRC', out_trg_lang='TGT', split='train'):
    """
    data_dir: input dir, contains directories for language pairs named l1-l2
    """
    os.makedirs(outdir, exist_ok=True)

    out_src_fname = '{}/{}.{}'.format(outdir, split, out_src_lang)
    out_trg_fname = '{}/{}.{}'.format(outdir, split, out_trg_lang)

    print('\n', out_src_fname)
    print(out_trg_fname)

    # concatenate train data
    if os.path.isfile(out_src_fname):
        os.unlink(out_src_fname)
    if os.path.isfile(out_trg_fname):
        os.unlink(out_trg_fname)

    for src_lang, trg_lang in (pbar := tqdm(lang_pair_list)):
        in_src_fname = '{}/{}-{}/{}.{}'.format(
            data_dir, src_lang, trg_lang, split, src_lang
        )
        in_trg_fname = '{}/{}-{}/{}.{}'.format(
            data_dir, src_lang, trg_lang, split, trg_lang
        )
        
        if not os.path.exists(in_src_fname):
            continue
        if not os.path.exists(in_trg_fname): 
            continue

        os.system('cat {} >> {}'.format(in_src_fname, out_src_fname))
        os.system('cat {} >> {}'.format(in_trg_fname, out_trg_fname))
        
        pbar.set_description(
            'src: {}, tgt: {}, in_src_fname: {}, in_trg_fname: {}'.format(
                src_lang, tgt_lang, in_src_fname, in_trg_fname
            )
        )

    corpus_stats(data_dir, outdir, lang_pair_list, split)


def corpus_stats(data_dir, outdir, lang_pair_list, split):
    """
    data_dir: input dir, contains directories for language pairs named l1-l2
    """

    with open('{}/{}_lang_pairs.txt'.format(outdir, split), 'w', encoding='utf-8') as lpfile:

        for src_lang, trg_lang in (pbar := tqdm(lang_pair_list)):
            print('src: {}, tgt:{}'.format(src_lang, trg_lang))

            in_src_fname = '{}/{}-{}/{}.{}'.format(
                data_dir, src_lang, trg_lang, split, src_lang)
            if not os.path.exists(in_src_fname):
                continue

            corpus_size = 0
            with open(in_src_fname, 'r', encoding='utf-8') as infile:
                corpus_size = sum(map(lambda x: 1, infile))

            lpfile.write('{}\t{}\t{}\n'.format(
                src_lang, trg_lang, corpus_size))
            pbar.set_description(in_src_fname)


if __name__ == '__main__':
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    src_lang = sys.argv[3]
    tgt_lang = sys.argv[4]
    split = sys.argv[6]
    languages_list = sys.argv[5]

    INDIC_LANGS = languages_list.split('+')
    lang_pair_list = [(['en', lang] if src_lang == 'en' else [lang, 'en']) for lang in INDIC_LANGS]

    concat_data(in_dir, out_dir, lang_pair_list, split=split)

