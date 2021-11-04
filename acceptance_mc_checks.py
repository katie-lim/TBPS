"""
This is to run some basic checks on the sim data so I can see what we are working with.
"""

import pandas as pd
import matplotlib.pyplot as plt
from data_tools import filter_dataset

particles = ['mu_plus_', 'mu_minus_', 'K_', 'Pi_']
p_col_names = ['P', 'PT', 'PE', 'PX', 'PY', 'PZ']

df_sim = pd.read_pickle('data/acceptance_mc.pkl').fillna(0)
df_sim_filtered = filter_dataset(df_sim)
df_tot = pd.read_pickle('data/total_dataset.pkl').fillna(0)
df_tot_filtered = filter_dataset(df_tot)

if True:
    fig, axs = plt.subplots(2, 2, figsize=[10,8])
    axs[0,0].hist(df_tot['costhetal'])
    axs[0,0].set_xlabel(r'cos($\theta_l$)')
    axs[0,0].set_ylabel('N')
    axs[0,0].set_title('Total dataset')

    axs[0,1].hist(df_sim['costhetal'])
    axs[0,1].set_xlabel(r'cos($\theta_l$)')
    axs[0,1].set_ylabel('N')
    axs[0,1].set_title('Simulated dataset')

    h, bins, patches = axs[1,0].hist(df_tot_filtered['costhetal'])
    axs[1,0].set_xlabel(r'cos($\theta_l$)')
    axs[1,0].set_ylabel('N')
    axs[1,0].set_title('Total dataset - filtered')

    axs[1,1].hist(df_sim_filtered['costhetal'])
    axs[1,1].set_xlabel(r'cos($\theta_l$)')
    axs[1,1].set_ylabel('N')
    axs[1,1].set_title('Simulated dataset - filtered')
    fig.tight_layout()


