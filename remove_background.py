
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from data_tools_and_filters import *





def remove_background(filtered_df, p=0.15,hist=True):
    '''
    Function meant reduce the background from a filtered total dataset.
    Trains a gradient boosting classifier on the simulated files for the 
    measurement and backgrounds. Classifies rows in measured dataset as
    background or not, keeping only the 'true' decay signal.
    
    

    Params
    -----
    filtered_df: dataframe with some filters applied
    p: float defining the cutoff probability for a row being discarded as background
    hist: boolean specifying whether histogram based gradient decent should be used

    Returns
    -----
    pandas.DataFrame with background removed
    
    '''
    
    
    sim_df = make_clean_df('data/sig.csv')
    
    background_files = ['jpsi.csv','psi2S.csv', 'jpsi_mu_k_swap.csv',
                        'jpsi_mu_pi_swap.csv','k_pi_swap.csv',
                        'phimumu.csv','pKmumu_piTok_kTop.csv',
                        'pKmumu_piTop.csv']
    background_dfs = [make_clean_df(file) for file in background_files]
    
    #Suggested training categories from have 'good predicting power'
    #https://hsf-training.github.io/analysis-essentials/advanced-python/30Classification.html#Using-a-classifier
    train_cat = ['mu_plus_PT', 'mu_minus_ETA', 
                     'mu_plus_MC15TuneV1_ProbNNmu','mu_minus_PT', 
                     'mu_minus_ETA', 'mu_minus_MC15TuneV1_ProbNNmu',
                     'K_PT', 'K_ETA', 'K_MC15TuneV1_ProbNNk','Pi_PT', 
                     'Pi_ETA', 'Pi_MC15TuneV1_ProbNNpi']
    
    
    sim_df['prob_sig'] = 1
    
    for df in background_dfs:
        df['prob_sig'] = 0
    
    #Merge simulated singal and background singal dataframes to train on 
    training_df = pd.concat([sim_df].extend(background_dfs), copy=True, ignore_index=True) 
    
    #Define Classifier, train using specified categories and training df
    if hist:
        BDT = HistGradientBoostingClassifier()
        BDT.fit(training_df[train_cat], training_df['prob_sig'])
    
    else:
        BDT = GradientBoostingClassifier()
        BDT.fit(training_df[train_cat], training_df['prob_sig'])
    
    #Use trained classifier to get probability a row in the total dataset is background
    prob_background = BDT.predict_proba(filtered_df[train_cat])[:,0]
    filtered_df['prob_background'] = prob_background
    
    #Remove rows with a high likelihood of being background and return
    return filtered_df[filtered_df['prob_background']<p]
   
    


        
    




        
    


