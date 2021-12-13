import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from data_tools_and_filters import *
from data_tools import *
from sklearn.ensemble import AdaBoostClassifier
from iminuit import Minuit
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

filtered_ML_df = make_clean_df("ML_filtered.pkl")
simulated_ML_df = make_clean_df("ML_simulated.pkl")

#probability of signal
prob_cut = 0.996
prob_cut_list = [0.9, 0.993, 0.9995, 0.9999, 0.98, 0.996, 0.993, 0.987, 0.993, 0.993]
#prob_cut_list = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

split_sim = split_into_bins(simulated_ML_df)
split_data = split_into_bins(filtered_ML_df)

for i in range(10):
    split_sim[i] = split_sim[i][split_sim[i]['prob_signal']>prob_cut_list[i]]
    split_data[i] = split_data[i][split_data[i]['prob_signal']>prob_cut_list[i]]
    
filtered = pd.concat(split_data, copy=True, ignore_index=True).drop_duplicates()
simulated = pd.concat(split_sim, copy=True, ignore_index=True).drop_duplicates() 

#Remove rows with a high likelihood of being background and return


#mass range filtering
filtered = filtered.loc[(filtered["B0_MM"] > 5170) & (filtered["B0_MM"]  < 5700)].reset_index(drop=True)
filtered = filtered.loc[(filtered["Kstar_MM"] > 795.9) & (filtered["Kstar_MM"]  < 995.9)].reset_index(drop=True)


#apply same filtered for simulated for acceptance function modelling

sim = simulated.loc[(simulated["B0_MM"] > 5170) & (simulated["B0_MM"]  < 5700)].reset_index(drop=True)
sim = simulated.loc[(simulated["Kstar_MM"] > 795.9) & (simulated["Kstar_MM"]  < 995.9)].reset_index(drop=True)


#%%
def get_acceptance_func_nosplit(number_of_bins_in_hist = 25):
    """
    Returns the acceptance function without bin splitting
    The acceptance function is obtained by fitting a 6th order polynomial to the filtered simulated dataset.
    """
    #simulated_dataset = load_dataset("acceptance_mc.pkl")
    #filtered_dataset = filter_dataset(simulated_dataset)
    
    filtered_dataset = sim
    
    h, bins = np.histogram(filtered_dataset["costhetal"], bins=number_of_bins_in_hist, density=True)
    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

    popt,cov = np.polyfit(bincenters, h, 6, cov=True)
    x = np.linspace(-1,1,50)
    p = np.poly1d(popt)
    return p

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

#inputs
filepath = "total_dataset.pkl"
fpath_si  = "std_predictions_si.json"
fpath_pi = "std_predictions_pi.json"
filtered_dataset = filtered


total_dataset = load_dataset(filepath)
fls_pred, afbs_pred, fl_errs_pred, afb_errs_pred = load_predictions(fpath_si, fpath_pi)

number_of_bins_in_hist = 25
data_bins = split_into_bins(filtered_dataset)

# Obtain the acceptance function for this bin
acceptance_func = get_acceptance_func_nosplit(number_of_bins_in_hist)
normalisation_factor = get_normalisation_factor(acceptance_func)
_test_bin = 3
_test_afb = 0.7
_test_fl = 0.0


bin_number_to_check = 8  # bin that we want to check in more details in the next cell
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

plt.figure(figsize=(8, 5))
plt.subplot(221)
bin_results_to_check.draw_mnprofile('afb', bound=20)
plt.subplot(222)
bin_results_to_check.draw_mnprofile('fl', bound=20)
plt.tight_layout()
plt.show()

bin_to_plot = 8
cos_theta_l_bin = data_bins[bin_to_plot]['costhetal']
plt.figure()
hist, _bins, _ = plt.hist(cos_theta_l_bin, bins=number_of_bins_in_hist)
x = np.linspace(-1, 1, number_of_bins_in_hist)
pdf_multiplier = np.sum(hist) * (np.max(cos_theta_l_bin) - np.min(cos_theta_l_bin)) / number_of_bins_in_hist
y = d2gamma_p_d2q2_dcostheta(fl=fls[bin_to_plot], afb=afbs[bin_to_plot], cos_theta_l=x, acceptance_func=acceptance_func, normalisation_factor=normalisation_factor) * pdf_multiplier
plt.plot(x, y, label=f'Fit for bin {bin_to_plot}')
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.legend()
plt.grid()
plt.show()

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
#plt.legend(bbox_to_anchor=(1.1, 1.05))
plt.legend()
plt.show()