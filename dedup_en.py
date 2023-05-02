import pandas as pd 
from sys import argv
from os import listdir 
from os import makedirs
from tqdm import tqdm
from collections import defaultdict

base_path = argv[1]
base_path_clean_wo_any_nway = argv[2]
k = float(argv[3])

total = 0
total_deduped = 0
final_en = []
en_count = {}
final_dict = defaultdict(list)
en_dict = defaultdict(list)


for lang_pair in sorted(listdir(base_path)):
    lang = lang_pair.split('-')[1]
    with open(f"{base_path}/{lang_pair}/labse.txt") as fs, \
         open(f"{base_path}/{lang_pair}/train.en") as fe, \
         open(f"{base_path}/{lang_pair}/train.{lang}", encoding='utf-8') as fl:
        labse = [float(x.strip()) for x in fs]
        en = [x.strip() for x in fe]
        indic = [x.strip() for x in fl]

    df = pd.DataFrame(list(zip(indic, en, labse)), columns =['indic', 'en', 'labse'])
    
    df = df.sort_values(by='labse', ascending=False)

    df = df.drop_duplicates(subset='en', keep='first')
    print(f"{lang} - after monolingual dedup: {len(df)/1e6}M", end=' | ')

    df = df.loc[df['labse'] > k]
    print(f"{lang} - after HQ filteration: {len(df)/1e6}M", end=' ')

    en = df['en'].tolist()
    indic = df['indic'].tolist()
    en_count[lang] = len(en)
    total += len(en)

    for (en_, indic_) in zip(en, indic):
        en_dict[en_].append((lang, indic_))
    
    print()

print(f"total: {total/1e6}M")

for (en_, translations) in tqdm(en_dict.items()):
    lang, indic_ = min(translations, key=lambda x: en_count[x[0]])
    final_dict[lang].append((en_, indic_))

for (lang, pairs) in final_dict.items():
    en, indic = zip(*pairs)
    total_deduped += len(en)
    print(f"{lang} - after crosslingual dedup: {len(en)/1e6}M")

    makedirs(f"{base_path_clean_wo_any_nway}/en-{lang}", exist_ok=True) 
    with open(f"{base_path_clean_wo_any_nway}/en-{lang}/train.en", 'w') as fe, \
         open(f"{base_path_clean_wo_any_nway}/en-{lang}/train.{lang}", 'w', encoding='utf-8') as fl:
        fe.write('\n'.join(en))
        fl.write('\n'.join(indic))

print(f"total_deduped: {total_deduped/1e6}M")
