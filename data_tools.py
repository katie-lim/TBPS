import numpy as np
import pandas as pd
import pickle5 as pickle


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
