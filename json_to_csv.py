from sys import argv
from os import listdir
import pandas as pd
import json

bleu_results = []
chrf2_results = []
base_path = "results"

for model in argv[1:]:
    for lang_pair in sorted(listdir(f"{base_path}/{model}")):
        lang = lang_pair.split('-')[1].split('.')[0]
        with open(f"{base_path}/{model}/{lang_pair}", 'r') as f:
            bleu_dict, chrf2_dict = json.load(f)
        
        bleu_results.append([lang, model, bleu_dict["score"]])
        chrf2_results.append([lang, model, chrf2_dict["score"]])

df_bleu = pd.DataFrame(bleu_results, columns=["lang", "model", "score"])
df_bleu = df_bleu.pivot(*df_bleu).rename_axis(columns=None).reset_index()
averages = df_bleu.iloc[:, 1:].mean(axis=0)
df_bleu = df_bleu.append(averages, ignore_index=True)
df_bleu.to_csv(f"{base_path}/bleu.csv", index=False)

df_chrf2 = pd.DataFrame(chrf2_results, columns=["lang", "model", "score"])
df_chrf2 = df_chrf2.pivot(*df_chrf2).rename_axis(columns=None).reset_index()
averages = df_chrf2.iloc[:, 1:].mean(axis=0)
df_chrf2 = df_chrf2.append(averages, ignore_index=True)
df_chrf2.to_csv(f"{base_path}/chrf2.csv", index=False)