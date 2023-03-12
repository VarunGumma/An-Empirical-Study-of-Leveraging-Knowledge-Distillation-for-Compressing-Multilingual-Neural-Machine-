from os import listdir, makedirs
from sys import argv
import numpy as np 

base_path = "../sam_v0.3/distilled_indic_en"
base_path_hq = "../sam_v0.3/HQ_distilled_indic_en"
k = float(argv[1])

for lang_pair in sorted(listdir(base_path)):
    l = lang_pair.split('-')[-1]
    with open(f"{base_path}/{lang_pair}/train.{l}", encoding='utf-8') as fl, \
         open(f"{base_path}/{lang_pair}/train.en") as fe:
        indic = np.array(fl.readlines(), dtype=object)
        en = np.array(fe.readlines(), dtype=object)
        labse = np.loadtxt(f"{base_path}/{lang_pair}/labse.txt")
        comet = np.loadtxt(f"{base_path}/{lang_pair}/comet.txt")

    u, s = labse.mean(), labse.std()
    print(lang_pair, u, s)

    msk = labse > (u + k*s)
    indic_hq, en_hq, labse_hq, comet_hq = indic[msk], en[msk], labse[msk], comet[msk]

    makedirs(f"{base_path_hq}/{lang_pair}", exist_ok=True)

    with open(f"{base_path_hq}/{lang_pair}/train.{l}", 'w', encoding='utf-8') as fl, \
         open(f"{base_path_hq}/{lang_pair}/train.en", 'w') as fe:
        fl.write(''.join(indic_hq))
        fe.write(''.join(en_hq))
        np.savetxt(f"{base_path_hq}/{lang_pair}/labse.txt", labse_hq, delimiter='\n')
        np.savetxt(f"{base_path_hq}/{lang_pair}/comet.txt", comet_hq, delimiter='\n')