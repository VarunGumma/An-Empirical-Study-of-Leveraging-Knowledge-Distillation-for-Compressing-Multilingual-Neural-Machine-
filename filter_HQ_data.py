from os import listdir, makedirs
from sys import argv
import numpy as np 

total, total_hq = 0, 0
base_path = argv[1]
base_path_hq = argv[2]
k = float(argv[3])

for lang_pair in sorted(listdir(base_path)):
    l = lang_pair.split('-')[-1]
    with open(f"{base_path}/{lang_pair}/train.{l}") as fl, \
         open(f"{base_path}/{lang_pair}/train.en") as fe:
        indic = np.array(fl.readlines(), dtype=object)
        en = np.array(fe.readlines(), dtype=object)
        labse = np.loadtxt(f"{base_path}/{lang_pair}/labse.txt", dtype=float)
        # comet = np.loadtxt(f"{base_path}/{lang_pair}/comet.txt", dtype=float)

    u, s = labse.mean(), labse.std()
    total += len(indic)

    msk = (labse > (u + k*s))
    # msk = labse > k
    # indic_hq, en_hq = indic[msk], en[msk]
    indic_hq, en_hq, labse_hq = indic[msk], en[msk], labse[msk] 
    # indic_hq, en_hq, labse_hq, comet_hq = indic[msk], en[msk], labse[msk], comet[msk]
    total_hq += len(indic_hq)

    print(lang_pair, len(indic_hq), len(indic), len(indic_hq)/len(indic))

    # makedirs(f"{base_path_hq}/{lang_pair}", exist_ok=True)

    # with open(f"{base_path_hq}/{lang_pair}/train.{l}", 'w') as fl, \
    #      open(f"{base_path_hq}/{lang_pair}/train.en", 'w') as fe:
        # fl.write(''.join(indic_hq))
        # fe.write(''.join(en_hq))
        # np.savetxt(f"{base_path_hq}/{lang_pair}/labse.txt", labse_hq, delimiter='\n')
        # np.savetxt(f"{base_path_hq}/{lang_pair}/comet.txt", comet_hq, delimiter='\n')

print('total', total, total_hq, total_hq/total)