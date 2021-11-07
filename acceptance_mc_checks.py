"""
This is to run some basic checks on the sim and actual data so I can see what we are working with.
"""

import matplotlib.pyplot as plt
from data_tools_and_filters import make_clean_df
from filtering_process import filters_and_params, filter_and_log_index, lhc_data_fpath, sim_data_fpath


lhc_df = make_clean_df(lhc_data_fpath)
sim_df = make_clean_df(sim_data_fpath)

lhc_log = filter_and_log_index(lhc_df, filter_process=filters_and_params, mode='cumulative')
sim_log = filter_and_log_index(sim_df, filter_process=filters_and_params, mode='cumulative')

lhc_df_filtered = lhc_df.loc[lhc_log['Final']]
sim_df_filtered = sim_df.loc[sim_log['Final']]

if True:
    fig, axs = plt.subplots(2, 2, figsize=[10, 8])
    axs[0, 0].hist(lhc_df['costhetal'])
    axs[0, 0].set_xlabel(r'cos($\theta_l$)')
    axs[0, 0].set_ylabel('N')
    axs[0, 0].set_title('Total dataset')

    axs[0, 1].hist(sim_df['costhetal'])
    axs[0, 1].set_xlabel(r'cos($\theta_l$)')
    axs[0, 1].set_ylabel('N')
    axs[0, 1].set_title('Simulated dataset')

    axs[1, 0].hist(lhc_df_filtered['costhetal'])
    axs[1, 0].set_xlabel(r'cos($\theta_l$)')
    axs[1, 0].set_ylabel('N')
    axs[1, 0].set_title('Total dataset - filtered')

    axs[1, 1].hist(sim_df_filtered['costhetal'])
    axs[1, 1].set_xlabel(r'cos($\theta_l$)')
    axs[1, 1].set_ylabel('N')
    axs[1, 1].set_title('Simulated dataset - filtered')
    fig.tight_layout()
