import sys
import concurrent.futures
from tqdm import tqdm


add_token = lambda sent, src_lang, tgt_lang, delimiter=' ': f"__src__{src_lang}__{delimiter}__tgt__{tgt_lang}__{delimiter}{sent}"


def generate_lang_tag_iterator(infname):
    with open(infname, 'r', encoding='utf-8') as infile:
        for line in infile:
            src, tgt, count = line.strip().split('\t')
            for _ in range(int(count)):
                yield (src, tgt)


def process_line(data):
    (l1, l2), src_sent, tgt_sent = data
    return add_token(src_sent.strip(), l1, l2) + '\n', tgt_sent.strip() + '\n'



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

        with concurrent.futures.ProcessPoolExecutor() as executor:

            # zip the iterators into a single iterator
            data = zip(lang_tag_iterator, srcfile, tgtfile)

            # iterate over results concurrently
            for result in tqdm(executor.map(process_line, data)):
                outsrcfile.write(result[0])
                outtgtfile.write(result[1])