import pandas as pd
from os import listdir, rename
from subprocess import check_output

comet_model = "wmt20-comet-qe-da-v2"
# src - En
# Hyp - Indic 

base_path = "../Downloads/data_raw"

print(f"using {comet_model}")

for dataset in sorted(listdir(base_path)):
    for lang_pair in sorted(listdir(f"{base_path}/{dataset}")):
        full_path = f"{base_path}/{dataset}/{lang_pair}"
        src, tgt = lang_pair.split('-')
        rename(f'{full_path}/scores.txt', f'{full_path}/labse.txt')
        
        cmd = f"comet-score -s {full_path}/train.{src} -t {full_path}/train.{tgt} --num_workers 32 --batch_size 128 --seed_everything 2023 --model {comet_model}"

        out = check_output(cmd, shell=True, text=True).strip().split('\n')[:-1]
        scores = [s.split('\t')[-1].split(': ')[-1] for s in out]
        
        with open(f"{full_path}/comet.txt", 'w') as f:
            f.write('\n'.join(scores))
