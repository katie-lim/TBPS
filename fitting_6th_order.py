import numpy as np
import matplotlib.pyplot as plt
from data_tools import load_dataset, filter_dataset

filepath = "data/acceptance_mc.pkl"

total_dataset = load_dataset(filepath)

# filter out data where we are not certain that the mu plus/minus are muons, the K is a kaon, etc
filtered_dataset = filter_dataset(total_dataset)

fig, axs = plt.subplots(2, 2, figsize=[10, 8])
axs = axs.flatten()

for i in range(4):
    nbins = 2 * i + 8
    h, bins, patches = axs[i].hist(filtered_dataset['costhetal'], bins=nbins, density=False)
    # centres = (bins[1:] + bins[:-1]) / 2
    bincenters = np.mean(np.vstack([bins[0:-1], bins[1:]]), axis=0)
    popt, cov = np.polyfit(bincenters, h, 6, cov=True)
    x = np.linspace(-1, 1, 50)
    p = np.poly1d(popt)
    axs[i].plot(x, p(x), label=np.array2string(popt, precision=2))
    axs[i].set_xlabel(r'$cos(\theta_l)$')
    axs[i].set_ylabel('Number of candidates')
    axs[i].set_title(f'{nbins} Bins')
    axs[i].grid()
    axs[i].legend(loc='lower center')

fig.tight_layout()
