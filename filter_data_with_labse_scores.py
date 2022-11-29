from sys import argv
from os import listdir, system

src_dir = argv[1]
dest_dir = argv[2]

system(f"rm -rf {dest_dir}")
system(f"mkdir {dest_dir}")

for lang_pair in listdir(src_dir):
    lang = lang_pair.split('-')[1]
    en_sents_hq, indic_sents_hq = [], []
    with open(f"{src_dir}/{lang_pair}/train.{lang}") as fi, \
         open(f"{src_dir}/{lang_pair}/train.en") as fe, \
         open(f"{src_dir}/{lang_pair}/scores.txt") as fs:
         indic_sents, en_sents, scores = fi.readlines(), fe.readlines(), fs.readlines()

    for (indic_sent, en_sent, score) in zip(indic_sents, en_sents, scores):
        if float(score) > 0.8:
            en_sents_hq.append(en_sent)
            indic_sents_hq.append(indic_sent)
    
    print(f"[INFO]\tfiltered {len(en_sents_hq)} out of {len(en_sents)} in {lang}")

    system(f"mkdir {dest_dir}/{lang_pair}")
    with open(f"{dest_dir}/{lang_pair}/train.{lang}", 'w') as fi, \
         open(f"{dest_dir}/{lang_pair}/train.en", 'w') as fe:
         fi.write(''.join(indic_sents_hq))
         fe.write(''.join(en_sents_hq))

    
        
