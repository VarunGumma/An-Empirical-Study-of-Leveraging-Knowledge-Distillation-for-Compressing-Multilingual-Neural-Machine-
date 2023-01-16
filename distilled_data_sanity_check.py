from random import sample
from subprocess import run
from os import listdir
from operator import itemgetter

base_path = "../data_raw/v2"

for dirname in listdir(base_path):
    lang = dirname.split('-')[1]
    with open(f"{base_path}/{dirname}/train.{lang}", 'r') as f:
        data = f.readlines()
    
    values = random.sample(list(enumerate(data)), 2500)
    indices = map(itemgetter(0), values)
    sents = map(itemgetter(1), values)

    with open(f"{base_path}/{dirname}/sample.{lang}", 'w') as f:
        f.write(''.join(sents))
    with open(f"{base_path}/{dirname}/sample.indices", 'w') as f:
        f.write('\n'.join(indices))