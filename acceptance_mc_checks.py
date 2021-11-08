"""
This is to run some basic checks on the sim and actual data so I can see what we are working with.
"""

import matplotlib.pyplot as plt
from data_tools_and_filters import *
from filtering_process import filter_and_log_index, lhc_data_fpath, sim_data_fpath

filter_process_1 = [[type_combined_filter, ('mu_plus', 0.3)],
                    [type_combined_filter, ('mu_minus', 0.3)],
                    [type_combined_filter, ('K', 0.2)],
                    [type_combined_filter, ('Pi', 0.3)],

                    [type_IPCHI2_OWNPV_filter, ('mu_plus', 9)],
                    [type_IPCHI2_OWNPV_filter, ('mu_minus', 9)],
                    [type_IPCHI2_OWNPV_filter, ('K', 9)],
                    [type_IPCHI2_OWNPV_filter, ('Pi', 9)],

                    [mu_pt_filter, ('mu_plus', 300)],
                    [mu_pt_filter, ('mu_minus', 300)],

                    [parent_ENDVERTEX_CHI2_filter, ('Kstar', 9)],
                    [parent_ENDVERTEX_CHI2_filter, ('B0', 9)],

                    [b0_IPCHI2_OWNPV_filter, 25],
                    [b0_FDCHI2_OWNPV_filter, 100],
                    [b0_DIRA_OWNPV_filter, 0.9995]]


def quick_setup(bin=10):
    lhc_df = q2_bin_filter(make_clean_df(lhc_data_fpath), range_int=bin)
    sim_df = q2_bin_filter(make_clean_df(sim_data_fpath), range_int=bin)

    lhc_log = filter_and_log_index(lhc_df, filter_process=filter_process_1, mode='cumulative')
    sim_log = filter_and_log_index(sim_df, filter_process=filter_process_1, mode='cumulative')

    lhc_df_filtered = lhc_df.loc[lhc_log['Final']]
    sim_df_filtered = sim_df.loc[sim_log['Final']]

    return lhc_df, lhc_df_filtered, sim_df, sim_df_filtered


def make_4_plot(param, actual_init, actual_final,
                simulated_init, simulated_final, nbins=20):
    if param == 'costhetal':
        xlabel = r'cos($\theta_l$)'
    elif param == 'costhetak':
        xlabel = r'cos($\theta_k$)'
    elif param == 'phi':
        xlabel = r'$\phi$'
    elif param == 'q2':
        xlabel = r'$q^2$ (MeV$^2$/$c^4$)'
    else:
        raise NameError('The param must be one of costhetal, costhetak, phi or q2')

    fig, axs = plt.subplots(2, 2, figsize=[10, 8])

    # Real data before filtering
    axs[0, 0].hist(actual_init[param], bins=nbins)
    axs[0, 0].set_xlabel(xlabel)
    axs[0, 0].set_ylabel('N')
    axs[0, 0].set_title('Total dataset')

    # Simulated data before filtering
    axs[0, 1].hist(simulated_init[param], bins=nbins)
    axs[0, 1].set_xlabel(xlabel)
    axs[0, 1].set_ylabel('N')
    axs[0, 1].set_title('Simulated dataset')

    # Real data after filtering
    axs[1, 0].hist(actual_final[param], bins=nbins)
    axs[1, 0].set_xlabel(xlabel)
    axs[1, 0].set_ylabel('N')
    axs[1, 0].set_title('Total dataset - filtered')

    # Simulated data after filtering
    axs[1, 1].hist(simulated_final[param], bins=nbins)
    axs[1, 1].set_xlabel(xlabel)
    axs[1, 1].set_ylabel('N')
    axs[1, 1].set_title('Simulated dataset - filtered')

    fig.tight_layout()
    return fig


def make_16_plot(actual_init, actual_final,
                 simulated_init, simulated_final, nbins=11, q2_bin=10):
    fig, axs = plt.subplots(4, 4, figsize=[11, 8])
    for i, param in enumerate(['costhetal', 'costhetak', 'phi', 'q2']):
        if param == 'costhetal':
            xlabel = r'cos($\theta_l$)'
        elif param == 'costhetak':
            xlabel = r'cos($\theta_k$)'
        elif param == 'phi':
            xlabel = r'$\phi$'
        elif param == 'q2':
            xlabel = r'$q^2$ (MeV$^2$/$c^4$)'

        x, y = 2 * (i // 2), 2 * (i % 2)
        # Real data before filtering
        axs[x + 0, y + 0].hist(actual_init[param], bins=nbins)
        axs[x + 0, y + 0].set_xlabel(xlabel)
        axs[x + 0, y + 0].set_title('Total dataset')

        # Simulated data before filtering
        axs[x + 0, y + 1].hist(simulated_init[param], bins=nbins)
        axs[x + 0, y + 1].set_xlabel(xlabel)
        axs[x + 0, y + 1].set_title('Simulated dataset')

        # Real data after filtering
        axs[x + 1, y + 0].hist(actual_final[param], bins=nbins)
        axs[x + 1, y + 0].set_xlabel(xlabel)
        axs[x + 1, y + 0].set_title('Total dataset - filtered')

        # Simulated data after filtering
        axs[x + 1, y + 1].hist(simulated_final[param], bins=nbins)
        axs[x + 1, y + 1].set_xlabel(xlabel)
        axs[x + 1, y + 1].set_title('Simulated dataset - filtered')

        fig.suptitle(rf'Histogram counts for bin {q2_bin}: {q2_bin_ranges[q2_bin]} MeV$^2$/$c^4$')
        fig.tight_layout(h_pad=0.08, w_pad=0.08)
    return fig


# lhc_df, lhc_df_filtered, sim_df, sim_df_filtered = quick_setup()

# To plot the costhetal (the cosine of the angle between the beam axis and the plane of muon decay)
# make_4_plot('costhetal', actual_init=lhc_df, actual_final=lhc_df_filtered,
#             simulated_init=sim_df, simulated_final=sim_df_filtered)

# To plot the costhetak (the cosine of the angle between the beam axis and the plane of kaon/pion decay)
# make_4_plot('costhetak', actual_init=lhc_df, actual_final=lhc_df_filtered,
#             simulated_init=sim_df, simulated_final=sim_df_filtered)

# To plot the phi (the difference in angle of the kaon/pion decay plane to the muon decay plane)
# make_4_plot('phi', actual_init=lhc_df, actual_final=lhc_df_filtered,
#             simulated_init=sim_df, simulated_final=sim_df_filtered)

# To plot the q^2
# make_4_plot('q2', actual_init=lhc_df, actual_final=lhc_df_filtered,
#             simulated_init=sim_df, simulated_final=sim_df_filtered)

for i in range(11):
    lhc_df, lhc_df_filtered, sim_df, sim_df_filtered = quick_setup(bin=i)
    fig = make_16_plot(actual_init=lhc_df, actual_final=lhc_df_filtered,
                       simulated_init=sim_df, simulated_final=sim_df_filtered, q2_bin=i)
    # fig.savefig(f'plots/bin_{i}_16_plot.png')
