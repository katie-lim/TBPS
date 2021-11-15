#%%
import numpy as np
import pandas as pd
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier
import matplotlib.pyplot as plt
from data_tools import *
from data_tools_and_filters import *



SM_sim = load_dataset('sig.pkl')

b1 = load_dataset('phimumu.pkl')
b2 = load_dataset('jpsi.pkl')
b3 = load_dataset('jpsi_mu_k_swap.pkl')
b4 = load_dataset('jpsi_mu_pi_swap.pkl')
b5 = load_dataset('k_pi_swap.pkl')
b6 = load_dataset('pKmumu_piTok_kTop.pkl')
b7 = load_dataset('pKmumu_piTop.pkl')
b8 = load_dataset('psi2S.pkl')

background = pd.concat([b1, b2, b3, b4, b5, b6, b7, b8])

measured_signal = load_dataset('total_dataset.pkl')

print("Data Loaded...")

columns_train = ['mu_plus_PT', 'mu_minus_ETA', 'mu_plus_MC15TuneV1_ProbNNmu','mu_minus_PT', 'mu_minus_ETA', 'mu_minus_MC15TuneV1_ProbNNmu',
                 'K_PT', 'K_ETA', 'K_MC15TuneV1_ProbNNk','Pi_PT', 'Pi_ETA', 'Pi_MC15TuneV1_ProbNNpi']
columns_train = ['mu_plus_PT', 'mu_plus_PID','mu_minus_PT', 'mu_minus_PID',
                 'K_PT', 'K_PID','Pi_PT', 'Pi_PID', 'q2', 'B0_IPCHI2_OWNPV']
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

SM_sim = PID(SM_sim, 'mu_plus')
SM_sim = PID(SM_sim, 'mu_minus')
SM_sim = PID(SM_sim, 'K')
SM_sim = PID(SM_sim, 'Pi')

print("Simulation PIDs Calcualted...")

background = PID(background, 'mu_plus')
background = PID(background, 'mu_minus')
background = PID(background, 'K')
background = PID(background, 'Pi')

print("Background PIDs Calcualted...")

measured_signal = PID(measured_signal, 'mu_plus')
measured_signal = PID(measured_signal, 'mu_minus')
measured_signal = PID(measured_signal, 'K')
measured_signal = PID(measured_signal, 'Pi')

print("Signal PIDs Calcualted...")

SM_sim['prob_sig'] = 1
background['prob_sig']=0
training_data = pd.concat([SM_sim, background], copy=True, ignore_index=True)
BDT = HistGradientBoostingClassifier()
BDT.fit(training_data[columns_train], training_data['prob_sig'])

print("GradientBoostingClassifier fitted...")

prob_background = BDT.predict_proba(measured_signal[columns_train])[:,0]
measured_signal['prob_background'] = prob_background
#%%

filtered = measured_signal[measured_signal['prob_background']>0.9]


plt.hist(filtered['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()
plt.figure()


plt.hist(measured_signal['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()

#%%
filtered2 = filter_dataset(filtered)
filtered3 = filter_dataset(load_dataset('total_dataset.pkl'))

plt.hist(filtered2['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()
plt.figure()
plt.hist(filtered3['costhetal'], bins=25, density=True)
plt.xlabel(r'$cos(\theta_l)$')
plt.ylabel(r'Number of candidates')
plt.grid()
plt.show()
# %%
