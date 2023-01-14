from sys import argv
from os import listdir
import pandas as pd
import json

bleu_results = []
chrf2pp_results = []

for model in argv[1:]:
    for benchmark in sorted(listdir(f"results/{model}")):
        for lang_pair in sorted(listdir(f"results/{model}/{benchmark}")):

            lang = lang_pair.split('-')[1]
            with open(f"results/{model}/{benchmark}/{lang_pair}/{lang}.json", 'r') as f:
                bleu_dict, chrf2pp_dict = json.load(f)
            
            bleu_results.append([f"{benchmark} - {lang}", model, bleu_dict["score"]])
            chrf2pp_results.append([f"{benchmark} - {lang}", model, chrf2pp_dict["score"]])

df_bleu = pd.DataFrame(bleu_results, columns=["benchmark", "model", "score"])
df_bleu = df_bleu.pivot(*df_bleu).rename_axis(columns=None).reset_index()
df_bleu.to_csv("bleu.csv", index=False)

df_chrf2pp = pd.DataFrame(chrf2pp_results, columns=["benchmark", "model", "score"])
df_chrf2pp = df_chrf2pp.pivot(*df_chrf2pp).rename_axis(columns=None).reset_index()
df_chrf2pp.to_csv("chrf2pp.csv", index=False)