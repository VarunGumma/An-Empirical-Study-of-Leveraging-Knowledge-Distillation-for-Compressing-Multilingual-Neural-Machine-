from sys import argv
from os import listdir
import pandas as pd
import json

bleu_results = []
chrf2_results = []
base_path = "results"

for model in argv[1:]:
    for benchmark in sorted(listdir(f"{base_path}/{model}")):
        for lang_pair in sorted(listdir(f"{base_path}/{model}/{benchmark}")):

            lang = lang_pair.split('-')[1]
            with open(f"{base_path}/{model}/{benchmark}/{lang_pair}/{lang}.json", 'r') as f:
                bleu_dict, chrf2_dict = json.load(f)
            
            bleu_results.append([f"{benchmark} - {lang}", model, bleu_dict["score"]])
            chrf2_results.append([f"{benchmark} - {lang}", model, chrf2_dict["score"]])

df_bleu = pd.DataFrame(bleu_results, columns=["benchmark", "model", "score"])
df_bleu = df_bleu.pivot(*df_bleu).rename_axis(columns=None).reset_index()
df_bleu.to_csv(f"{base_path}/bleu.csv", index=False)

df_chrf2 = pd.DataFrame(chrf2_results, columns=["benchmark", "model", "score"])
df_chrf2 = df_chrf2.pivot(*df_chrf2).rename_axis(columns=None).reset_index()
df_chrf2.to_csv(f"{base_path}/chrf2.csv", index=False)