import matplotlib.pyplot as plt 
import numpy as np
from os import listdir

base_path = "../Downloads/data/v2"
fig, ax = plt.subplots(3, 4, figsize=(25, 25))

ylims = [0.6e4, 5e4, 3e4, 6e4, 3.5e4, 4.5e4, 3.5e4, 1.2e4, 2.5e4, 4e4, 4e4]

for idx, (dirname, ylim) in enumerate(zip(sorted(listdir(base_path)), ylims)):
    try:
        with open(f"{base_path}/{dirname}/og_distilled_sim_scores.txt", 'r') as f:
            data = np.array([float(x) for x in f.readlines()])
        print(f"{dirname} -- {data.mean():.3f} -- {data.std():.3f}")
        i, j = idx // 4, idx % 4
        ax[i, j].hist(
            data, 
            bins="auto"
        )
        ax[i, j].set_title(
            f"{dirname}\nmean: {data.mean():.3f}\n std: {data.std():.3f}", 
            fontsize=25
        )
        ax[i, j].axvline(
            data.mean(), 
            linestyle='--', 
            linewidth=2.0, 
            color="red"
        )
        ax[i, j].set_ylim(0, ylim)
        ax[i, j].xaxis.set_tick_params(labelsize=20)
        ax[i, j].yaxis.set_tick_params(labelsize=20)
    except FileNotFoundError:
        pass

fig.tight_layout()
plt.savefig("img1.png", dpi=600)