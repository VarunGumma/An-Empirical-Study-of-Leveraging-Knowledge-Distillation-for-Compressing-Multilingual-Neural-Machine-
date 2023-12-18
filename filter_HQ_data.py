from os import listdir
from sys import argv
import numpy as np 
import os

total, total_hq = 0, 0
base_path = argv[1]
base_path_hq = argv[2]
k = float(argv[3])

for lang_pair in sorted(listdir(base_path)):
    l = lang_pair.split('-')[-1]
    with open(os.path.join(base_path, lang_pair, f"train.{l}")) as fl, \
         open(os.path.join(base_path, lang_pair, "train.en")) as fe:
        indic = np.array(fl.readlines(), dtype=object)
        en = np.array(fe.readlines(), dtype=object)
        labse = np.loadtxt(os.path.join(base_path, lang_pair, "labse.txt"), dtype=float)

    u, s = labse.mean(), labse.std()
    total += len(indic)

    msk = (labse > (u + k*s))
    indic_hq, en_hq, labse_hq = indic[msk], en[msk], labse[msk] 
    total_hq += len(indic_hq)
    print(lang_pair, len(indic_hq), len(indic), len(indic_hq)/len(indic))

print('total', total, total_hq, total_hq/total)