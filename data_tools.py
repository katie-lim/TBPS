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



def filter_dataset_by_particles(df, prob=0.95):
    """
    Filters the DataFrame by only keeping the rows where we are confident that the mu plus and mu minus are indeed muons, the K is indeed a kaon, and the pi is a pion.

    The cutoff probability is defined by the parameter called prob.

    Params
    ------
    df: pandas.DataFrame containing the data
    prob: float defining the cutoff probability at which we keep/discard rows (set to 0.95 by default)

    Returns
    ------
    pandas.DataFrame containing the filtered data
    """

    # Filter the dataset as described above
    filtered_df = df[(df["mu_plus_MC15TuneV1_ProbNNmu"] > prob) & (df["mu_minus_MC15TuneV1_ProbNNmu"] > prob) & (df["K_MC15TuneV1_ProbNNk"] > prob) & (df["Pi_MC15TuneV1_ProbNNpi"] > prob)]

    return filtered_df


def filter_dataset_by_common_vertex(df, sig_level=0.95):
    """
    Filters the DataFrame by removing rows where the particles do not meet at a vertex.

    This is decided based on the B0_ENDVERTEX_CHI2 value and the chosen significance level, set by the parameter sig_level.

    (Also adds another column to the DataFrame, called B0_ENDVERTEX_p_value.)

    Params
    ------
    df: pandas.DataFrame containing the data
    sig_level: float defining the significance level at which we keep/discard rows (set to 0.95 by default)

    Returns
    ------
    pandas.DataFrame containing the filtered data
    """

    # Calculate the p-values / find the probability that each set of particles forms a vertex. Store the p-values in a new column.
    def calc_p_value(row):
        return chi2.sf(row.B0_ENDVERTEX_CHI2, row.B0_ENDVERTEX_NDOF)

    filtered_df = df.copy()
    filtered_df["B0_ENDVERTEX_p_value"] = filtered_df.apply(calc_p_value, axis=1)


    # Only keep candidates where the p-value is below our significance level (i.e. there is a high probability the particles form a vertex)
    filtered_df = filtered_df[filtered_df["B0_ENDVERTEX_p_value"] > sig_level]

    # (we could do a similar analysis for Kstar_ENDVERTEX_CHI2 and J_psi_ENDVERTEX_CHI2?)

    return filtered_df