# -*- coding: utf-8 -*-
"""
Created on Wed Nov 10 21:05:06 2021

@author: Daryl
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from iminuit import Minuit
from data_tools import *
from acceptance_func import *
import json



def load_predictions (fpath_si, fpath_pi):
    """
    Returns 4 lists of afb and fl predictions

    Parameters
    ----------
    fpath_si : filepath of the si predictions
    fpath_pi : filepath of the pi predictions

    """
    pd.set_option("display.latex.repr",False)
    si_preds = {}
    pi_preds = {}
    with open(fpath_si,"r") as _f:
        si_preds = json.load(_f)
    with open(fpath_pi,"r") as _f:
        pi_preds = json.load(_f)
    
    si_list = []
    for _binNo in si_preds.keys():
        si_frame = pd.DataFrame(si_preds[_binNo]).transpose()
        si_list.append(si_frame)
    pi_list = []
    for _binNo in pi_preds.keys():
        pi_frame = pd.DataFrame(pi_preds[_binNo]).transpose()
        pi_list.append(pi_frame)
    fls_pred = []
    afbs_pred = []
    fl_errs_pred = []
    afb_errs_pred = []
    for predictions in si_list:
        fls_pred.append(predictions['val'][0])
        afbs_pred.append(predictions['val'][1])
        fl_errs_pred.append(predictions['err'][0])
        afb_errs_pred.append(predictions['err'][1])
    
    return fls_pred, afbs_pred, fl_errs_pred, afb_errs_pred


def d2gamma_p_d2q2_dcostheta(fl, afb, cos_theta_l, acceptance_func, normalisation_factor):
    """
    Returns the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param cos_theta_l: cos(theta_l)
    :return:
    """
    ctl = cos_theta_l
    c2tl = 2 * ctl ** 2 - 1
    acceptance = 1/acceptance_func(cos_theta_l) # 1/acceptance_func because we want to invert the warping
    scalar_array = 3/8 * (3/2 - 1/2 * fl + 1/2 * c2tl * (1 - 3 * fl) + 8/3 * afb * ctl) * acceptance
    #normalised_scalar_array = scalar_array * 2  # normalising scalar array to account for the non-unity acceptance function
    normalised_scalar_array = scalar_array/normalisation_factor
    return normalised_scalar_array

def log_likelihood(fl, afb, bin_number):
    """
    Returns the negative log-likelihood of the pdf defined above
    :param fl: f_l observable
    :param afb: a_fb observable
    :param _bin: number of the bin to fit
    :return:
    """
    bin_number = int(bin_number)
    _bin = data_bins[bin_number]
    ctl = _bin['costhetal']


    normalised_scalar_array = d2gamma_p_d2q2_dcostheta(fl=fl, afb=afb, cos_theta_l=ctl, acceptance_func=acceptance_func, normalisation_factor=normalisation_factor)
    return - np.sum(np.log(normalised_scalar_array))

def get_acceptance_func_nosplit(number_of_bins_in_hist = 25):
    """
    Returns the acceptance function without bin splitting
    The acceptance function is obtained by fitting a 6th order polynomial to the filtered simulated dataset.
    """
    simulated_dataset = load_dataset("data/acceptance_mc.pkl")
    filtered_dataset = filter_dataset(simulated_dataset)

    h, bins = np.histogram(filtered_dataset["costhetal"], bins=number_of_bins_in_hist, density=True)
    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

    popt,cov = np.polyfit(bincenters, h, 6, cov=True)
    x = np.linspace(-1,1,50)
    p = np.poly1d(popt)
    return p


filepath = "data/total_dataset.pkl"
total_dataset = load_dataset(filepath)

#finds minimum over a 2d search grid
#range of values to be searched
search_arr1 = np.arange(8.7, 9.1, 0.1)
search_arr2 = np.arange(0.9999, 1, 0.00001)


fpath_si  = "predictions/std_predictions_si.json"
fpath_pi = "predictions/std_predictions_pi.json"

fls_pred, afbs_pred, fl_errs_pred, afb_errs_pred = load_predictions(fpath_si, fpath_pi)



error_arr = []

for params in search_arr1:
    error_arr2 = []
    for params2 in search_arr2:
        
        #search grid set to ipchi2 of B0 and dira angle
        filtered_dataset = filter_dataset(total_dataset, B0_ipcs_opv_lim = params, B0_dira = params2)
        
        
        #mass range that we study
        filtered_dataset = filtered_dataset.loc[(filtered_dataset["B0_MM"] > 5170) & (filtered_dataset["B0_MM"]  < 5700)].reset_index(drop=True)
        number_of_bins_in_hist = 25
        data_bins = split_into_bins(filtered_dataset)
    
        # Obtain the acceptance function for this bin
        acceptance_func = get_acceptance_func_nosplit(number_of_bins_in_hist)
        normalisation_factor = get_normalisation_factor(acceptance_func)
        _test_bin = 3
        _test_afb = 0.7
        _test_fl = 0.0
        

        bin_number_to_check = 0  # bin that we want to check in more details in the next cell
        bin_results_to_check = None
        
        log_likelihood.errordef = Minuit.LIKELIHOOD
        decimal_places = 3
        starting_point = [-0.1,0.0]
        fls, fl_errs = [], []
        afbs, afb_errs = [], []
        for i in range(len(data_bins)):
            m = Minuit(log_likelihood, fl=starting_point[0], afb=starting_point[1], bin_number=i)
            m.fixed['bin_number'] = True  # fixing the bin number as we don't want to optimize it
            m.limits=((-1.0, 1.0), (-1.0, 1.0), None)
            m.migrad()
            m.hesse()
            if i == bin_number_to_check:
                bin_results_to_check = m
            fls.append(m.values[0])
            afbs.append(m.values[1])
            fl_errs.append(m.errors[0])
            afb_errs.append(m.errors[1])
            #print(f"Bin {i}: {np.round(fls[i], decimal_places)} pm {np.round(fl_errs[i], decimal_places)},", f"{np.round(afbs[i], decimal_places)} pm {np.round(afb_errs[i], decimal_places)}. Function minimum considered valid: {m.fmin.is_valid}")

        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3), dpi=100)
        ax1.errorbar(np.linspace(0, len(fls_pred) - 1, len(fls_pred)), fls_pred, yerr=fl_errs_pred, fmt='o', markersize=2, label=r'$F_L$', color='red')
        ax2.errorbar(np.linspace(0, len(afbs_pred) - 1, len(afbs_pred)), afbs_pred, yerr=afb_errs_pred, fmt='o', markersize=2, label='Predicted', color='red')
        ax1.errorbar(np.linspace(0, len(fls) - 1, len(fls)), fls, yerr=fl_errs, fmt='o', markersize=2, label=r'$F_L$', color='blue')
        ax2.errorbar(np.linspace(0, len(afbs) - 1, len(afbs)), afbs, yerr=afb_errs, fmt='o', markersize=2, label='Observed', color='blue')
        ax1.grid()
        ax2.grid()
        ax1.set_ylabel(r'$F_L$')
        ax2.set_ylabel(r'$A_{FB}$')
        ax1.set_xlabel(r'Bin number')
        ax2.set_xlabel(r'Bin number')
        plt.tight_layout()
        plt.legend(bbox_to_anchor=(1.1, 1.05))
        plt.title("Search: {} , {}".format(round(params, 6), round(params2, 6)))
        plt.show()
        
        #minimising the error between observed and predicted
        error_arr2.append(np.sum(np.abs(np.array(fls_pred)-np.array(fls))+np.abs(np.array(afbs_pred)-np.array(afbs))))
    error_arr.append(error_arr2)
    
    
search_arr1_index = int(str((np.argmin(error_arr)+1)/len(error_arr[0]))[0])
search_arr2_index = int(round(((np.argmin(error_arr)+1)/len(error_arr[0]) - search_arr1_index)*len(error_arr[0]) - 1, 1))

#prints out the vars that give minimum error
print("Minimum in Search Array 1 (axis = 0): {}".format(search_arr1[search_arr1_index]))
print("Minimum in Search Array 2 (axis = 1): {}".format(search_arr2[search_arr2_index]))