from sys import argv
from os import listdir
import pandas as pd
import json

def write_df(results, score_name):
    df = pd.DataFrame(results, columns=["lang", "model", score_name])
    df = df.pivot(*df).rename_axis(columns=None).reset_index()
    averages = df.iloc[:, 1:].mean(axis=0)
    df = df.append(averages, ignore_index=True)
    df.to_csv(f"{base_path}/{score_name}.csv", index=False)



bleu_results = []
chrf2_results = []
comet_results = []
base_path = argv[1]

for model in argv[2:]:
    for lang_pair in sorted(listdir(f"{base_path}/{model}")):
        extn = lang_pair.split('.')[1]
        
        if extn == "json":
            lang = lang_pair.split('-')[1].split('.')[0]
            with open(f"{base_path}/{model}/{lang_pair}", 'r') as f:
                bleu_dict, chrf2_dict = json.load(f)
            bleu_results.append([lang, model, bleu_dict["score"]])
            chrf2_results.append([lang, model, chrf2_dict["score"]])
        elif extn == "txt":
            lang = lang_pair.split('_')[0].split('-')[1]
            with open(f"{base_path}/{model}/{lang_pair}", 'r') as f:
                line = [line.strip() for line in f][0].split('\t')[1]
                score = float(line.split(': ')[1])
                comet_results.append([lang, model, score])
                
write_df(bleu_results, "bleu")
write_df(chrf2_results, "chrf++")
write_df(comet_results, "comet")