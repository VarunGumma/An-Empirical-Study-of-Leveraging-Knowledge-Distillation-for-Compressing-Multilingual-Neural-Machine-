import matplotlib.pyplot as plt 
import pandas as pd

BASE_PATH = "results"

df_dev = pd.read_csv(f"{BASE_PATH}/benchmark_scores.dev.csv")
df_test = pd.read_csv(f"{BASE_PATH}/benchmark_scores.test.csv")

L, width = len(df_dev.columns)-1, 0.2

fig, ax = plt.subplots(nrows=1, ncols=L)
plt.figure(figsize=(10, 20))

for i, name in enumerate(df_dev.columns[1:]):
    ax[0].