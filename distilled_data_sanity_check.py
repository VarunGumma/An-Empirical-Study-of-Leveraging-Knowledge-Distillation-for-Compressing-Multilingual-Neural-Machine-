from random import sample
from os import listdir
from operator import itemgetter

base_path = "../data_raw/v2"

# for dirname in listdir(base_path):
#     lang = dirname.split('-')[1]
#     with open(f"{base_path}/{dirname}/train.{lang}", 'r') as f:
#         data_indic = f.readlines()
#     with open(f"{base_path}/{dirname}/train.en", 'r') as f:
#         data_en = f.readlines()

#     indices = range(len(data_en))
    
#     values = sample(list(zip(indices, data_indic, data_en)), 3000)
#     indices = map(itemgetter(0), values)
#     sents_indic = map(itemgetter(1), values)
#     sents_en = map(itemgetter(2), values)

#     with open(f"{base_path}/{dirname}/sample.{lang}", 'w') as f:
#         f.write(''.join(sents_indic))
#     with open(f"{base_path}/{dirname}/sample.en", 'w') as f:
#         f.write(''.join(sents_en))
#     with open(f"{base_path}/{dirname}/sample_indices.txt", 'w') as f:
#         f.write('\n'.join([str(i) for i in indices]))

for dirname in listdir(base_path):
    lang = dirname.split('-')[1]
    with open(f"{base_path}/{dirname}/sample_indices.txt", 'r') as f:
        indices = [int(i) for i in f.readlines()]
    with open(f"{base_path}_distilled_indic_en/{dirname}/train.en", 'r') as f:
        data_en = f.readlines()
        sents = [data_en[i] for i in indices]
    with open(f"{base_path}/{dirname}/sample_distil.en", 'w') as f:
        f.write(''.join(sents))