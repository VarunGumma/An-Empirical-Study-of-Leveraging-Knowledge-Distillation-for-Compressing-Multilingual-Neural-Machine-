import sys
from tqdm import tqdm

def add_token(sent, src_lang, tgt_lang, delimiter=' '):
    """ add special tokens specified by tag_infos to each element in list

    sent: untagged input sentence
    src_lang: source sentence code
    tgt_lang: target sentence code

    output is a sentence with source and target language tags of the form "__src__{src_lang} __tgt__{tgt_lang} {sent}"

    """

    return f"__src__{src_lang}{delimiter}__tgt__{tgt_lang}{delimiter}{sent}"


def generate_lang_tag_iterator(infname):
    with open(infname, 'r', encoding='utf-8') as infile:
        for line in infile:
            src, tgt, count = line.strip().split('\t')
            for _ in range(int(count)):
                yield (src, tgt)


if __name__ == '__main__':

    expdir = sys.argv[1]
    dset = sys.argv[2]

    src_fname = f'{expdir}/bpe/{dset}.SRC'
    tgt_fname = f'{expdir}/bpe/{dset}.TGT'
    out_src_fname = f'{expdir}/final/{dset}.SRC'
    out_tgt_fname = f'{expdir}/final/{dset}.TGT'
    meta_fname = f'{expdir}/data/{dset}_lang_pairs.txt'
    
    lang_tag_iterator = generate_lang_tag_iterator(meta_fname)

    with open(src_fname, 'r', encoding='utf-8') as srcfile, \
         open(tgt_fname, 'r', encoding='utf-8') as tgtfile, \
         open(out_src_fname, 'w', encoding='utf-8') as outsrcfile, \
         open(out_tgt_fname, 'w', encoding='utf-8') as outtgtfile:

        for ((l1, l2), src_sent, tgt_sent) in tqdm(zip(lang_tag_iterator, srcfile, tgtfile)):
            outsrcfile.write(add_token(src_sent.strip(), l1, l2) + '\n')
            outtgtfile.write(tgt_sent.strip() + '\n')
