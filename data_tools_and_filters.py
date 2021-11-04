import pandas as pd

decay_particles = ['mu_plus', 'mu_minus', 'K', 'Pi']
prob_particles = ['k', 'pi', 'mu', 'e', 'p']
net = '_MC15TuneV1_ProbNN'

def make_clean_df(fpath):
    """
    Reads a pkl or csv file into a dataframe
    Crops negative probability values to zero
    Fills NaN values with zeros
    Parameters
    ----------
    fpath: The filepath to be opened, either a .pkl or a .csv

    Returns: A pandas dataframe
    -------
    """
    if fpath[-3:] == 'pkl':
        df = pd.read_pickle(fpath)
    elif fpath[-3:] == 'csv':
        df = pd.read_csv(fpath)
    else:
        raise NameError('Can only read .pkl and .csv files')
    df.fillna(0, inplace=True)
    for col in df.columns:
        if net in col:
            # Force the values to be probabilities if they are negative
            df[col].clip(0, 1)
    return df

def type_check(type):
    """
    Formats the input type (a decay particle) into the correct column name string, or raises an error
    """
    if type in decay_particles:
        return type
    elif type == 'mu+':
        return 'mu_plus'
    elif type == 'mu-':
        return 'mu_minus'
    elif type == 'k':
        return 'K'
    elif type == 'pi':
        return 'Pi'
    else:
        raise ValueError(f'Incorrect type, must be one of: {decay_particles}')

def type_combined_filter(df, type, lower_bound=0.2):
    """
    Filters a df based on a specific particle: mu+, mu-, K or Pi.
    Runs through all the columns for this particle and multiplies together:
     - The correct identification probability, i.e. when the particle type matches
     - (1 - the mistaken identification probability), for every other particle type
    The types are k, pi, mu, e, and p.
    The resulting dataframe has therefore only been filtered once:
    The valid rows are where this multiplication is > lower_bound

    Parameters
    ----------
    df - The input dataframe
    type - The type of particle we are filtering by
    lower_bound - The minimum value that correct_prob*(1-misID_prob)**4 can take,
    for each of the 4 misID probabilities in a given row

    Returns
    -------
    The filtered dataframe
    """
    type = type_check(type)
    df['cumprod'] = 1
    for part in prob_particles:
        col_name = type + net + part
        if part in type.lower():
            # Correct identification probability
            df['cumprod'] *= df[col_name]
        else:
            # Mistaken identification probability
            df['cumprod'] *= 1 - df[col_name]

    return df[df['cumprod'] > lower_bound].drop(columns='cumprod')

def type_IPCHI2_OWNPV_filter(df, type, PKMuMu_OWNPV_filter=9):
    """
    Filters a dataframe based on if a specific particle type meets this condition:
     - its IPCHI2_OWNPV > PKMuMu_OWNPV_filter
    Parameters
    ----------
    df - the input df
    type - which particle we are filtering based off: one of mu_plus, mu_minus, K, or Pi
    PKMuMu_OWNPV_filter - The lower bound for this particle's IPCHI2_OWNPV to be valid

    Returns - A filtered dataframe
    -------

    """
    type = type_check(type)
    colname = type + "_IPCHI2_OWNPV"
    return df[df[colname] > PKMuMu_OWNPV_filter]

def mu_pt_filter(df, type, mu_pt_lim=300):
    type = type_check(type)
    if type not in ['mu_plus', 'mu_minus']:
        print(f'Not filtering on {type}\'s transverse momentum!')
        return df
    return df[df[type + '_PT'] > mu_pt_lim]

def parent_ENDVERTEX_CHI2_filter(df, parent, ratio=9):
    if parent not in ['Kstar', 'B0']:
        raise NameError('Parent must be Kstar or B0')
    return df[df[parent+'_ENDVERTEX_CHI2']/df[parent+'_ENDVERTEX_NDOF'] < ratio]

def b0_IPCHI2_OWNPV_filter(df, B0_ipcs_opv_lim=25):
    return df[df['B0_IPCHI2_OWNPV'] < B0_ipcs_opv_lim]

def b0_FDCHI2_OWNPV_filter(df, B0_fdcs_opv_lim=100):
    return df[df['B0_FDCHI2_OWNPV'] > B0_fdcs_opv_lim]

def b0_DIRA_OWNPV_filter(df, B0_dira=0.9995):
    return df[df['B0_DIRA_OWNPV'] > B0_dira]


# Below are old filters kept here just in case
'''
def type_intersect_filter(df, type, correct_thresh=0.95, misID_thresh=0.1):
    """
    Filters a df based on a specific particle: mu+, mu-, K or Pi.
    Runs through all the columns for this particle and checks that:
     - The correct identification probability is > correct_thresh
     - The mistaken identification probability is < misID_thresh
    The resulting dataframe is the intersection of all valid rows after each step.
    Parameters
    ----------
    df - input dataframe to be filtered
    type - the particle we want to filter based on
    correct_thresh - the lower bound for a correctly identified probability
    misID_thresh - the upper bound for a mistakenly identified probability

    Returns
    -------
    A filtered dataframe based on the given parameters
    """
    if type not in decay_particles:
        raise ValueError(f'Incorrect type, must be one of: {decay_particles}')
    for part in prob_particles:
        name = type + net + part
        if part in type.lower():
            # Correct identification probability
            df = df[df[name] > correct_thresh]
        else:
            # Mistaken identification probability
            df = df[df[name] < misID_thresh]
    return df

def mu_plus_prob_filter(df, prob=0.95):
    """
    Checking that mu+ is actually a mu
    """
    return df[df['mu_plus_MC15TuneV1_ProbNNmu'] > prob]

def mu_minus_prob_mu_filter(df, prob=0.95):
    """
    Checking that mu- is actually a mu
    """
    return df[df['mu_minus_MC15TuneV1_ProbNNmu'] > prob]

def k_prob_k_filter(df, prob=0.95):
    """
    Checking that k is actually a k
    """
    return df[df['mu_plus_MC15TuneV1_ProbNNmu'] > prob]

def pi_prob_pi_filter(df, prob=0.95):
    """
    Checking that pi is actually a pi
    """
    return df[df['mu_plus_MC15TuneV1_ProbNNmu'] > prob]

def filter_dataset(df, prob=0.3, PKMuMu_OWNPV_filter = 9, K_s_ratio = 9, mu_pt_lim = 300, \
                                B0_dira = 0.9995, B0_ratio = 9, B0_ipcs_opv_lim = 25, B0_fdcs_opv_lim = 100):
    """
    Filters the DataFrame by only keeping the rows where we are confident that the mu plus and mu minus are indeed muons, the K is indeed a kaon, and the pi is a pion.
    The cutoff probability is defined by the parameter called prob.
    
    It subsequently filters the data to ensure the daughter particles truly came from the decay vertex, and that the B0 came from the primary vertex.
    The paramaters were taken from this academic paper: http://www.ep.ph.bham.ac.uk/publications/thesis/lxp_thesis.pdf


    Params
    ------
    df: pandas.DataFrame containing the data
    prob: float defining the cutoff probability at which we keep/discard rows (set to 0.3 by default, approx equal to 0.8^5)
    PKMuMu_OWNPV_filter: Defines the cutoff value for the IPCHI2_OWNPV values of the data for the: Kaon, pion, and muons.
    K_s_ratio: Defines the cutoff value for the Kstar_ENDVERTEX_CHI2/Kstar_ENDVERTEX_NDOF.
    mu_pt_lim: Defines the cutoff value for mu_plus_PT and mu_minus_PT - the muon transverse momentum in MeV/c^2.
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
    column_list_Pi = ["Pi_MC15TuneV1_ProbNNpi", "Pi_MC15TuneV1_ProbNNk","Pi_MC15TuneV1_ProbNNmu", "Pi_MC15TuneV1_ProbNNp", "Pi_MC15TuneV1_ProbNNe"]
    
    column_list = [column_list_mu_plus, column_list_mu_minus, column_list_K, column_list_Pi]
    
    pf_df = df.copy()
    #I have structured the column lists such that when put through this for loop the elements are in the appropriate column
    for i in column_list:
        #set -ve probabilities to 0
        pf_df[i[0]][pf_df[i[0]] < 0] = 0
        pf_df[i[1]][pf_df[i[1]] < 0] = 0
        pf_df[i[2]][pf_df[i[2]] < 0] = 0
        pf_df[i[3]][pf_df[i[3]] < 0] = 0
        pf_df[i[4]][pf_df[i[4]] < 0] = 0
        pf_df = pf_df[((pf_df[i[0]]) * (1 - pf_df[i[1]]) * (1 - pf_df[i[2]]) * (1 - pf_df[i[3]]) * (1 - pf_df[i[4]])) > prob]




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

'''
