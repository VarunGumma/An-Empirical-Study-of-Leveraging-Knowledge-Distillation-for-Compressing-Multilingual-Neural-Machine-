from random import seed, random
from os import system, listdir
from tqdm import tqdm
from sys import argv
seed(1000000007)

BASE_PATH = argv[1]

for p in tqdm([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1], total=9):
    # p denotes percentage of original targets present in the mixed sample
    idx = int(p*100)
    system(f"cp -r {BASE_PATH} {BASE_PATH}_{idx}")
    for pair in listdir(f"{BASE_PATH}"):
            with open(f"{BASE_PATH}/{pair}/train.en", 'r') as f_org, \
                open(f"{BASE_PATH}_distilled/{pair}/train.en", 'r') as f_distil, \
                open(f"{BASE_PATH}_{idx}/{pair}/train.en", 'w') as f_mix:
                
                f_mix.write(
                    ''.join(
                        [
                            (l_org if random() <= p else l_distil) 
                            for (l_org, l_distil) in zip(f_org.readlines(), f_distil.readlines())
                        ]
                    )
                )

system(f"mv {BASE_PATH} {BASE_PATH}_100")
system(f"mv {BASE_PATH}_distilled {BASE_PATH}_0")