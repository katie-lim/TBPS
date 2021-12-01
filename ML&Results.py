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

K_mass = 493.677

def PID(df, type):
    type = type_check(type)
    df[type + '_PID'] = 1
    for part in prob_particles:
        col_name = type + net + part
        if part == 'e' or part == 'p':
            # Mistaken identification probability: electrons and protons are NOT what we are checking for
            df[type + '_PID'] *= 1 - df[col_name]
        elif part in type.lower():
            # Correct identification probability: decay particle is the type we are checking for
            df[type + '_PID'] *= df[col_name]
        else:
            # Mistaken identification probability: decay particle is NOT the type we are checking for
            df[type + '_PID'] *= 1 - df[col_name]

    return df

def modify_df(df):
    
    #re-defining columns for more accurate training
    df = PID(df, 'mu_plus')
    df = PID(df, 'mu_minus')
    df = PID(df, 'K')
    df = PID(df, 'Pi')
    df['Kstar_ENDVERTEX'] = df['Kstar_ENDVERTEX_CHI2']/df['Kstar_ENDVERTEX_NDOF']
    df['J_psi_ENDVERTEX'] = df['J_psi_ENDVERTEX_CHI2']/df['J_psi_ENDVERTEX_NDOF']
    df['B0_ENDVERTEX'] = df['B0_ENDVERTEX_CHI2']/df['B0_ENDVERTEX_NDOF']
    
    #reconstruct pion as kaon to give phi
    #calculate energy component in 4 four momentum
    df['phi_PX'] = df['Pi_PX']+df['K_PX']
    df['phi_PY'] = df['Pi_PY']+df['K_PY']
    df['phi_PZ'] = df['Pi_PZ']+df['K_PZ']
    df['phi_P'] = np.sqrt(df['phi_PX']*df['phi_PX']+
                          df['phi_PY']*df['phi_PY']+
                          df['phi_PZ']*df['phi_PZ'])
    df['Pi_K_PE'] = np.sqrt(K_mass*K_mass + df['Pi_P']*df['Pi_P'])
    df['phi_MM'] = np.sqrt((df['Pi_K_PE']+df['K_PE'])**2-(df['phi_P'])**2)
    
    #remove rows with reconstructed mass close to phi 
    df = df.loc[(df["phi_MM"] < 1000) | (df["phi_MM"]  > 1080)].reset_index(drop=True)
    
    #removed peaking backgrounds
    df = df.loc[(df["q2"] < 8) | (df["q2"]  > 11)].reset_index(drop=True)
    df = df.loc[(df["q2"] < 12.5) | (df["q2"]  > 15)].reset_index(drop=True)
    df = df.loc[(df["q2"] < 0.98) | (df["q2"]  > 1.1)].reset_index(drop=True)
    
    return df

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
    #simulated_dataset = load_dataset("acceptance_mc.pkl")
    #filtered_dataset = filter_dataset(simulated_dataset)
    
    filtered_dataset = sim
    
    h, bins = np.histogram(filtered_dataset["costhetal"], bins=number_of_bins_in_hist, density=True)
    bincenters = np.mean(np.vstack([bins[0:-1],bins[1:]]), axis=0)

    popt,cov = np.polyfit(bincenters, h, 6, cov=True)
    x = np.linspace(-1,1,50)
    p = np.poly1d(popt)
    return p


def get_acceptance_func(bin_number, number_of_bins_in_hist = 25):
    """
    Returns the acceptance function for a particular bin.
    The acceptance function is obtained by fitting a 6th order polynomial to the filtered simulated dataset.
    """

    simulated_dataset = load_dataset("data/acceptance_mc.pkl")


    data_bins = split_into_bins(simulated_dataset)
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

filtered_df = make_clean_df('data/total_dataset.pkl')
#filtered_df = filter_dataset(filtered_df)
simulated_df = make_clean_df('data/acceptance_mc.pkl')

filtered_df = modify_df(filtered_df)
simulated_df = modify_df(simulated_df)

   
sig_df = make_clean_df('data/sig.pkl')
sig_df = PID(sig_df, 'mu_plus')
sig_df = PID(sig_df, 'mu_minus')
sig_df = PID(sig_df, 'K')
sig_df = PID(sig_df, 'Pi')
sig_df['Kstar_ENDVERTEX'] = sig_df['Kstar_ENDVERTEX_CHI2']/sig_df['Kstar_ENDVERTEX_NDOF']
sig_df['J_psi_ENDVERTEX'] = sig_df['J_psi_ENDVERTEX_CHI2']/sig_df['J_psi_ENDVERTEX_NDOF']
sig_df['B0_ENDVERTEX'] = sig_df['B0_ENDVERTEX_CHI2']/sig_df['B0_ENDVERTEX_NDOF']


#remaining background files
background_files = ['jpsi_mu_k_swap.pkl',
                    'jpsi_mu_pi_swap.pkl','k_pi_swap.pkl',
                    'pKmumu_piTok_kTop.pkl',
                    'pKmumu_piTop.pkl']

background_dfs = [make_clean_df(file) for file in background_files]


#Suggested training categories from have 'good predicting power'
#https://hsf-training.github.io/analysis-essentials/advanced-python/30Classification.html#Using-a-classifier



#columns used in training
train_cat = ['mu_plus_PID',  'mu_plus_PT', 'mu_plus_ETA', 'mu_plus_IPCHI2_OWNPV',
       'mu_minus_PID', 'mu_minus_PT', 'mu_minus_ETA', 'mu_minus_IPCHI2_OWNPV',
       'K_PID','K_PT', 'K_ETA','K_IPCHI2_OWNPV', 'Pi_MC15TuneV1_ProbNNk',
       'Pi_PID', 'Pi_PT', 'B0_MM',
       'Pi_ETA', 'Pi_IPCHI2_OWNPV', 'B0_ENDVERTEX', 'B0_FDCHI2_OWNPV', 'Kstar_MM',
       'Kstar_ENDVERTEX', 'J_psi_MM', 'J_psi_ENDVERTEX', 'B0_IPCHI2_OWNPV',
       'B0_DIRA_OWNPV','B0_FD_OWNPV']


sig_df['prob_sig'] = 1

background_dfs2 = []
for df in background_dfs:
    df['prob_sig'] = 0
    #df.loc[(df["B0_MM"] > 5500) & (df["B0_MM"]  < 6000)].reset_index(drop=True)
    df = PID(df, 'mu_plus')
    df = PID(df, 'mu_minus')
    df = PID(df, 'K')
    df = PID(df, 'Pi')
    df['Kstar_ENDVERTEX'] = df['Kstar_ENDVERTEX_CHI2']/df['Kstar_ENDVERTEX_NDOF']
    df['J_psi_ENDVERTEX'] = df['J_psi_ENDVERTEX_CHI2']/df['J_psi_ENDVERTEX_NDOF']
    df['B0_ENDVERTEX'] = df['B0_ENDVERTEX_CHI2']/df['B0_ENDVERTEX_NDOF']
    background_dfs2.append(df)

    
    

#Merge simulated singal and background singal dataframes to train on 
df_list = [sig_df]
df_list.extend(background_dfs2)
training_df = pd.concat(df_list, copy=True, ignore_index=True) 

#Define Classifier, train using specified categories and training df
BDT = HistGradientBoostingClassifier(max_depth = 5, max_iter = 100, learning_rate = 0.1, min_samples_leaf = 5, max_leaf_nodes= 5)
#BDT = AdaBoostClassifier(n_estimators = 40, learning_rate = 1)
BDT.fit(training_df[train_cat], training_df['prob_sig'])

#Use trained classifier to get probability a row in the total dataset is signal
prob_signal= BDT.predict_proba(filtered_df[train_cat])[:,1]
filtered_df['prob_signal'] = prob_signal

prob_signal = BDT.predict_proba(simulated_df[train_cat])[:,1]
simulated_df['prob_signal'] = prob_signal


#%%

#probability of signal
prob_cut = 0.992

#Remove rows with a high likelihood of being background and return
filtered = filtered_df[filtered_df['prob_signal']>prob_cut]

#mass range filtering
filtered = filtered.loc[(filtered["B0_MM"] > 5170) & (filtered["B0_MM"]  < 5700)].reset_index(drop=True)
filtered = filtered.loc[(filtered["Kstar_MM"] > 795.9) & (filtered["Kstar_MM"]  < 995.9)].reset_index(drop=True)

#apply same filtered for simulated for acceptance function modelling
sim = simulated_df[simulated_df['prob_signal']>prob_cut]
sim = sim.loc[(sim["B0_MM"] > 5170) & (sim["B0_MM"]  < 5700)].reset_index(drop=True)
sim = sim.loc[(sim["Kstar_MM"] > 795.9) & (sim["Kstar_MM"]  < 995.9)].reset_index(drop=True)

#%%
#inputs
filepath = "data/total_dataset.pkl"
fpath_si  = "predictions/std_predictions_si.json"
fpath_pi = "predictions/std_predictions_pi.json"
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





