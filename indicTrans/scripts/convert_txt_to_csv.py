import pandas as pd
from sys import argv

def read(fname):
    with open(fname, 'r') as f:
        data = [line.strip() for line in f.readlines() if len(line.strip()) > 0]
    return [line.split(':')[1][:-1] for line in data if line.startswith("\"score\"")]

df = pd.DataFrame([read(f"../results/{ext}.txt") for ext in argv[1:]])
df.T.to_csv("../results/all.csv", header=argv[1:], index=False)