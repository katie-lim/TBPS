import numpy as np
import matplotlib.pyplot as plt
from data_tools_and_filters import *
from filtering_process import filter_and_log_index, filters_and_params


def poly_fit(df, col='costhetal', order=6, density=True):
    h, bins = np.histogram(df[col], bins='auto', density=density)
    bincenters = np.mean(np.vstack([bins[:-1], bins[1:]]), axis=0)
    popt, cov = np.polyfit(bincenters, h, order, cov=True)
    return np.poly1d(popt), h, bins

def norm_fit(simulated_unfiltered_df, simulated_filtered_df, actual_filtered_df,
             col='costhetal', order=6, plot=True):
    p0, h0, b0 = poly_fit(simulated_unfiltered_df, col, order, True)
    p1, h1, b1 = poly_fit(simulated_filtered_df, col, order, True)
    p2, h2, b2 = poly_fit(actual_filtered_df, col, order, True)

    xx = np.linspace(min(b0[0], b1[0], b2[0]), max(b0[-1], b1[-1], b2[-1]), 100)

    unwarped = p2(xx) * (p0(xx) / p1(xx))
    popt3, cov3 = np.polyfit(xx, unwarped, order, cov=True)
    p3 = np.poly1d(popt3)

    if plot:
        fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=[10, 7])
        ax1.bar(b0[:-1], h0, align='edge', width=b0[1:]-b0[:-1])
        ax1.plot(xx, p0(xx), label='Sim #nofilter fit')
        ax1.bar(b1[:-1], h1, align='edge', width=b1[1:]-b1[:-1])
        ax1.plot(xx, p1(xx), label='Sim filter fit')
        ax2.bar(b2[:-1], h2, align='edge', width=b2[1:]-b2[:-1])
        ax2.plot(xx, p2(xx), label='Actual fit')
        ax2.plot(xx, p3(xx), label='Unwarped?')
        fig.legend()
    return p3

if __name__ == '__main__':
    lhc_data_fpath = 'data/total_dataset.pkl'
    sim_data_fpath = 'data/acceptance_mc.pkl'

    lhc_df = make_clean_df(lhc_data_fpath)
    sim_df = make_clean_df(sim_data_fpath)

    lhc_log = filter_and_log_index(lhc_df, filter_process=filters_and_params, mode='cumulative')
    sim_log = filter_and_log_index(sim_df, filter_process=filters_and_params, mode='cumulative')

    lhc_filtered = lhc_df.loc[lhc_log['Final']]
    sim_filtered = sim_df.loc[sim_log['Final']]

    acceptance = norm_fit(simulated_unfiltered_df=sim_df, simulated_filtered_df=sim_filtered,
                          actual_filtered_df=lhc_filtered, col='costhetal', order=6, plot=True)
