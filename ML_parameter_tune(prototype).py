import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from data_tools_and_filters import *










def tune_params(classifier,df,max_iter=300,max_depth=300,hist=True):
    '''
    Intended to be used after remove_background
    Function takes trained classifier and df and:
        1) Decides on the most important training columns and then 
           retrains the classifier
        2) Optimises and retrains the classifier based on the most 
           preferable values of max_iter and max_depth
    
    

    Params
    -----
    classifier: trained sklearn classifier
    df: dataframe of merged data from all the files with the probabilities 
    hist: boolean specifying whether histogram based gradient decent has been used
    
    Returns
    -----
    retrained classifier
    
    '''
    
    
    
    
    #Get importances of each feature i.e column in the data file. Add up to 1
    feature_importance = classifier.feature_importances_
    
    
    important_indexes = []
    for i,feature in enumerate(feature_importance):
        if feature >= 1/(len(df.index)):
            important_indexes.append(i)
    
    
    column_names = np.array(df.keys())      
    training_columns = column_names[important_indexes]
    
    
    
    
    #Train new classifier on newly found good columns
    
    if hist:
        new_classifier = HistGradientBoostingClassifier()
        new_classifier.fit(df[training_columns], df['prob_sig'])
    
    else:
        new_classifier = GradientBoostingClassifier()
        new_classifier.fit(df[training_columns], df['prob_sig'])
    
    
    
    
    #List of number of iterations (i.e number of trees) to try
    pos_max_iter = np.arange(1,max_ter,10)
    
    #List of tree depths to try
    pos_max_depth = np.arange(1,max_depth,10)

    
    #Get new classifier with optimised parameters for depth and iterations
    params = {'max_depth':pos_max_depth,'max_iter':pos_max_iter}
    new_classifier = RandomizedSearchCV(new_classifier, params)
    
    return new_classifier
    
    
    
    
    
   
    
    
    
    

