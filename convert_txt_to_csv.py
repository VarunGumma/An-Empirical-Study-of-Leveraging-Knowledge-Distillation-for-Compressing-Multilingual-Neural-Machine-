import pandas as pd
from sys import argv

BASE_PATH = "results"
FNAMES = argv[1:]


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


df = pd.DataFrame([read_benchmark_scores(f"{BASE_PATH}/{fname}.txt") for fname in FNAMES]).T
df.columns = FNAMES

for (i, col) in enumerate(FNAMES[1:], 1):
    df.insert(i*2, f"scoredrop-{col}", df["indicTrans"]-df[col])
df.insert(0, "benchmarks", read_benchmark_names(f"{BASE_PATH}/{FNAMES[0]}.txt"))
    
df.to_csv(f"{BASE_PATH}/benchmark_scores.csv", index=False)