from sys import argv
import numpy as np
from os import listdir, system

src_dir = argv[1]
dest_dir = argv[2]
src_lang = argv[3]
tgt_lang = argv[4]

system(f"rm -rf {dest_dir}")
system(f"mkdir {dest_dir}")
total_len, hq_len = 0, 0

for lang_pair in listdir(src_dir):
    if src_lang == "en":
        tgt_lang = lang_pair.split('-')[1]
    else:
        src_lang = lang_pair.split('-')[1]
    en_sents_hq, indic_sents_hq = [], []
    with open(f"{src_dir}/{lang_pair}/train.{src_lang}") as fi, \
         open(f"{src_dir}/{lang_pair}/train.{tgt_lang}") as fe, \
         open(f"{src_dir}/{lang_pair}/scores.txt") as fs:
         indic_sents, en_sents, scores = fi.readlines(), fe.readlines(), np.float_(fs.readlines())

         u, std = np.mean(scores), np.std(scores)
         en_sents_hq = [en_sent for (en_sent, s) in zip(en_sents, scores) if (s > (u + 0.75*std))]
         indic_sents_hq = [indic_sent for (indic_sent, s) in zip(indic_sents, scores) if (s > (u + 0.75*std))]
    
    print(f"[INFO]\tfiltered {len(en_sents_hq)} out of {len(en_sents)} in {lang_pair}")
    hq_len += len(indic_sents_hq)
    total_len += len(indic_sents)

    system(f"mkdir {dest_dir}/{lang_pair}")
    with open(f"{dest_dir}/{lang_pair}/train.{src_lang}", 'w') as fi, \
         open(f"{dest_dir}/{lang_pair}/train.{tgt_lang}", 'w') as fe:
         fi.write(''.join(indic_sents_hq))
         fe.write(''.join(en_sents_hq))

print(f"aggregate: filtered {hq_len/1000000}M out of {total_len/1000000}M")

    
        
