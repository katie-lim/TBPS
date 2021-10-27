import numpy as np
import pandas as pd
import pickle5 as pickle
from scipy.stats import chi2


def load_dataset(filepath):
    """
    Function meant to load data from pickle file and impute NaNs with 0s (might be changed in future)

    Params
    -----

    filepath: str with the filepath

    Returns
    -----
    pandas.DataFrame with the pickle values
    """
    df = pd.read_pickle(filepath)

    return df.fillna(0)

def separate_dataset(df, columns_idx, prob=0.95):
    """
    Separates the DataFrame based on the probability columns into dataframe for each particle

    Params
    ------
    df: pandas.DataFrame with probability columns
    columns_idx: list with indexes of the columns containing probabilities
    prob: float defining the cutoff probability (set to 0.95 as default)

    Returns
    ------
    list of pandas.DataFrames for each column (has the same length as columns_idx)
    """
    separated_df = []
    for column in columns_idx:
        particle_df = df[df[df.columns[column]]>prob]
        separated_df.append(particle_df)
    return separated_df



def filter_dataset(df, prob=0.95, PKMuMu_OWNPV_filter = 9, K_s_ratio = 9, mu_pt_lim = 300, \
                                B0_dira = 0.9995, B0_ratio = 9, B0_ipcs_opv_lim = 25, B0_fdcs_opv_lim = 100):
    """
    Filters the DataFrame by only keeping the rows where we are confident that the mu plus and mu minus are indeed muons, the K is indeed a kaon, and the pi is a pion.
    The cutoff probability is defined by the parameter called prob.
    
    It subsequently filters the data to ensure the daughter particles truly came from the decay vertex, and that the B0 came from the primary vertex.
    The paramaters were taken from this academic paper: http://www.ep.ph.bham.ac.uk/publications/thesis/lxp_thesis.pdf


    Params
    ------
    df: pandas.DataFrame containing the data
    prob: float defining the cutoff probability at which we keep/discard rows (set to 0.95 by default)
    PKMuMu_OWNPV_filter: Defines the cutoff value for the IPCHI2_OWNPV values of the data for the: Kaon, pion, and muons.
    K_s_ratio: Defines the cutoff value for the Kstar_ENDVERTEX_CHI2/Kstar_ENDVERTEX_NDOF.
    mu_pt_lim: Defines the cutoff value for mu_plus_PT and mu_minus_PT - the muon total momentum in MeV/c^2.
    B0_dira: Defines the cutoff value for B0_DIRA_OWNPV - the cosine between the momentum and flight vectors.
    B0_ratio: Defines the cutoff value for B0_ENDVERTEX_CHI2/B0_ENDVERTEX_NDOF.
    B0_ipcs_opv_lim: Defines the cutoff value of B0_IPCHI2_OWNPV.
    B0_fdcs_opv_lim: Defines the cutoff value of B0_FDCHI2_OWNPV.
    

    Returns
    ------
    pandas.DataFrame containing the filtered data
    """

    # Produces Probability Filtered Data Frame (pf_df)
    column_list_mu_plus = ["mu_plus_MC15TuneV1_ProbNNmu", "mu_plus_MC15TuneV1_ProbNNk", "mu_plus_MC15TuneV1_ProbNNpi", "mu_plus_MC15TuneV1_ProbNNe", "mu_plus_MC15TuneV1_ProbNNp"]
    column_list_mu_minus = ["mu_minus_MC15TuneV1_ProbNNmu", "mu_minus_MC15TuneV1_ProbNNk", "mu_minus_MC15TuneV1_ProbNNpi", "mu_minus_MC15TuneV1_ProbNNe", "mu_minus_MC15TuneV1_ProbNNp"]
    column_list_K = ["K_MC15TuneV1_ProbNNk","K_MC15TuneV1_ProbNNmu", "K_MC15TuneV1_ProbNNpi", "K_MC15TuneV1_ProbNNe", "K_MC15TuneV1_ProbNNp"]
    column_list_Pi = ["Pi_MC15TuneV1_ProbNNp", "Pi_MC15TuneV1_ProbNNk","Pi_MC15TuneV1_ProbNNmu", "Pi_MC15TuneV1_ProbNNpi", "Pi_MC15TuneV1_ProbNNe"]
    
    column_list = [column_list_mu_plus, column_list_mu_minus, column_list_K, column_list_Pi]
    
    
    #I have structured the column lists such that when put through this for loop the elements are in the appropriate column
    for i in column_list:
        pf_df = df[((df[i[0]]) * (1 - df[i[1]]) * (1 - df[i[2]]) * (1 - df[i[3]])) > prob]




    #Uses momentum and chi^2 values to filter particles. These values were taken from the paper: http://www.ep.ph.bham.ac.uk/publications/thesis/lxp_thesis.pdf
    filtered_df = pf_df[(pf_df["Pi_IPCHI2_OWNPV"] > PKMuMu_OWNPV_filter)\
                        & (pf_df["K_IPCHI2_OWNPV"] > PKMuMu_OWNPV_filter)\
                            & (pf_df["mu_plus_IPCHI2_OWNPV"] > PKMuMu_OWNPV_filter)\
                            & (pf_df["mu_minus_IPCHI2_OWNPV"] > PKMuMu_OWNPV_filter)\
                            & (pf_df["mu_minus_IPCHI2_OWNPV"] > PKMuMu_OWNPV_filter)\
                                & (((pf_df["Kstar_ENDVERTEX_CHI2"])/(pf_df["Kstar_ENDVERTEX_NDOF"])) > K_s_ratio)\
                                    & (pf_df["mu_plus_PT"] > mu_pt_lim)\
                                        & (pf_df["mu_minus_PT"] > mu_pt_lim)\
                                            & (pf_df["B0_DIRA_OWNPV"] > B0_dira)\
                                                & (((pf_df["B0_ENDVERTEX_CHI2"])/(pf_df["B0_ENDVERTEX_NDOF"])) < B0_ratio)\
                                                    & (pf_df["B0_IPCHI2_OWNPV"] < B0_ipcs_opv_lim)\
                                                         & (pf_df["B0_FDCHI2_OWNPV"] > B0_fdcs_opv_lim)]

    return filtered_df


def filter_dataset_by_pt(df, p_ratio = 0.1):
    """
    Filters the DataFrame by removing rows with particles that have small transverse momentum
    
    Size of the transverse momentum is quantified by taking the ratio of the transverse momentum to its momentum in the z direction
    
    Particles that have a smaller ratio than p_ratio are removed from the dataset
    
    Params
    ------
    df: pandas.DataFrame containing the data
    sig_level: float defining the cutoff ratio at which we keep/discard rows (set to 0.1 by default)
    
    Returns
    ------
    pandas.DataFrame containing the filtered data
    """
    
    particles = ["mu_plus", "mu_minus", "K", "Pi"]
    
    filtered_df = df.copy()
    
    #iterate ratio calculation over all particles
    for particle in particles:
        filtered_df["{}_p_ratio".format(particle)] = filtered_df["{}_PT".format(particle)]/filtered_df["{}_PZ".format(particle)]
        filtered_df = filtered_df[filtered_df["{}_p_ratio".format(particle)] > p_ratio]
        
    return filtered_df
