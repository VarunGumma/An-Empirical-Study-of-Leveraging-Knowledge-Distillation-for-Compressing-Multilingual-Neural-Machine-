import pandas as pd
from sys import argv

BASE_PATH = "results"
split = argv[1]
FNAMES = argv[2:]


def read_benchmark_scores(fname):
    with open(fname, 'r') as f:
        data = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return [float(line.split(':')[1][:-1]) for line in data if line.startswith("\"score\"")]


def read_benchmark_names(fname):
    is_benchmark = lambda x: (x.startswith("wat") or \
                              x.startswith("wmt") or \
                              x.startswith("ufal") or \
                              x.startswith("pmi") or \
                              x.startswith("flores")) 

    with open(fname, 'r') as f:
        data = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return [line.split(':')[0] for line in data if is_benchmark(line)]


df = pd.DataFrame([read_benchmark_scores(f"{BASE_PATH}/{fname}.{split}.txt") for fname in FNAMES]).T
df.columns = FNAMES

df.insert(0, "benchmarks", read_benchmark_names(f"{BASE_PATH}/{FNAMES[0]}.{split}.txt"))
df.to_csv(f"{BASE_PATH}/benchmark_scores.{split}.csv", index=False)