import numpy as np
from data_tools import *
from scipy import integrate

def get_acceptance_func(bin_number, number_of_bins_in_hist = 25):
    """
    Returns the acceptance function for a particular bin.

    The acceptance function is obtained by fitting a 6th order polynomial to the filtered simulated dataset.
    """

    simulated_dataset = load_dataset("data/acceptance_mc.pkl")

    filtered_dataset = filter_dataset(simulated_dataset)

    data_bins = split_into_bins(filtered_dataset)
    bin = data_bins[bin_number]

    h, bins = np.histogram(bin["costhetal"], bins=number_of_bins_in_hist, density=True)
    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

    popt,cov = np.polyfit(bincenters, h, 6, cov=True)
    x = np.linspace(-1,1,50)
    p = np.poly1d(popt)

    return p


def get_normalisation_factor(p):
    area = integrate.quad(p, -1, 1)[0]

    return 2/area # not sure this is correct