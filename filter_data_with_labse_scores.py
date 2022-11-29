from sys import argv
from os import listdir, system

src_dir = argv[1]
dest_dir = argv[2]

system(f"rm -rf {dest_dir}")
system(f"mkdir {dest_dir}")

for lang_pair in listdir(src_dir):
    lang = lang_pair.split('-')[1]
    en_sents, indic_sents = [], []
    with open(f"{src_dir}/{lang_pair}/train.{lang}") as fi, \
         open(f"{src_dir}/{lang_pair}/train.en") as fe, \
         open(f"{src_dir}/{lang_pair}/scores.txt") as fs:
         for (indic_sent, en_sent, score) in zip(fi.readlines(), fe.readlines(), fs.readlines()):
            if float(score) > 0.8:
                en_sents.append(en_sent)
                indic_sents.append(indic_sent)

    system(f"mkdir {dest_dir}/{lang_pair}")
    with open(f"{dest_dir}/{lang_pair}/train.{lang}", 'w') as fi, \
         open(f"{dest_dir}/{lang_pair}/train.en", 'w') as fe:
         fi.write(''.join(indic_sents))
         fe.write(''.join(en_sents))

    
        
