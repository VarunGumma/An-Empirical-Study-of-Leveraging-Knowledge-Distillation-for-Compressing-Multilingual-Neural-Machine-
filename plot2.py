import matplotlib.pyplot as plt 
import numpy as np
from os import listdir

base_path = "../Downloads/data/v2_distilled_indic_en"
fig, ax = plt.subplots(3, 4, figsize=(25, 25))

for idx, dirname in enumerate(sorted(listdir(base_path))):
    try:
        with open(f"{base_path}/{dirname}/scores.txt", 'r') as f:
            data = np.array([float(x) for x in f.readlines()])
        print(f"{dirname} -- {data.mean():.3f} -- {data.std():.3f}")
        mean, std = data.mean(), data.std()
        hq_data = data[data > (mean + 0.75*std)]
        hq_percent = len(hq_data)/len(data)
        i, j = idx // 4, idx % 4
        ax[i, j].hist(
            data, 
            bins="auto"
        )
        ax[i, j].set_title(
            f"{dirname}\nmean: {data.mean():.3f}\nstd: {data.std():.3f}\nHQ%: {hq_percent:.3f}", 
            fontsize=25
        )
        ax[i, j].axvline(
            data.mean(), 
            linestyle='--', 
            linewidth=2.0, 
            color="red"
        )
        ax[i, j].xaxis.set_tick_params(labelsize=20)
        ax[i, j].yaxis.set_tick_params(labelsize=20)
    except FileNotFoundError:
        pass

fig.tight_layout()
plt.savefig("img2.png", dpi=600)