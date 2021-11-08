"""
This script will be the filtering process for a dataset.
This is to ensure that the same steps are applied to the simulated data as to the real data.

The filter_and_log_df function takes a dataframe and a filter_process and returns a dictionary.
Each entry has a string representation of the function with its parameters that was called on the dataframe
as its key, and the resulting filtered dataframe as its value.

The filter_process is a list of two-element lists.
I give filters_and_params as an example.
The first element is the name of the function to apply to the dataframe.
The second element is the parameter to pass to the function.
If there is more than one, they can be packed together into a tuple.

The mode can be set to cumulative, in which case each filter is applied to the
result of the previous step, or it can be set to individual, in which case each filter
is applied to the original dataframe that was passed in.

The filter_and_log_index function works almost exactly the same, but instead logs the INDEXES of the rows
that pass each filtering step. The dataframe is re-indexed each time so as to only have one df in memory,
rather than multiple dfs with very similar information.
Hence for an index log i_log, to get the actual dataframe after a filtering step with function f and
parameters p = (p0, p1, ...) , do:
df.loc[i_log['f(df, *p)']]
"""
from data_tools_and_filters import *

lhc_data_fpath = 'data/total_dataset.pkl'
sim_data_fpath = 'data/acceptance_mc.pkl'

filters_and_params = [[type_combined_filter, ('mu_plus', 0.3)],
                      [type_combined_filter, ('mu_minus', 0.3)],
                      [type_combined_filter, ('K', 0.2)],
                      [type_combined_filter, ('Pi', 0.3)],

                      [type_IPCHI2_OWNPV_filter, ('mu_plus', 9)],
                      [type_IPCHI2_OWNPV_filter, ('mu_minus', 9)],
                      [type_IPCHI2_OWNPV_filter, ('K', 9)],
                      [type_IPCHI2_OWNPV_filter, ('Pi', 9)],

                      [mu_pt_filter, ('mu_plus', 300)],
                      [mu_pt_filter, ('mu_minus', 300)],

                      [parent_ENDVERTEX_CHI2_filter, ('Kstar', 9)],
                      [parent_ENDVERTEX_CHI2_filter, ('B0', 9)],

                      [b0_IPCHI2_OWNPV_filter, 25],
                      [b0_FDCHI2_OWNPV_filter, 100],
                      [b0_DIRA_OWNPV_filter, 0.9995]]

def filter_and_log_df(df, filter_process, mode='cumulative'):
    """
    Takes in a dataframe, and a filter_process - a list of lists.
    The filter_process should have the names of functions to sequentially apply
    to the dataframe as the first entry, and the parameters for each function as the second.
    Multiple parameters can be passed in as a tuple.
    At each step, the resulting dataframe is pushed to an output dictionary.
    The key for the step is the string representation of the function that was called,
    WITH ITS PARAMETERS!
    This means we can track exactly which steps are cutting the most data.
    We can also save a specific filtering process because all the steps were passed to this function.
    Parameters
    ----------
    df: The input dataframe to be filtered
    filter_process: A list containing a series of 2 element lists.
    The first entry is the NAME of the function
    The second is the PARAMETERS to pass to it as its values (in a tuple if necessary).
    mode: Cumulative means each filter is applied to the result of the previous step.
    Individual means each filter is applied to the raw df that was passed in initially.

    Returns
    -------
    A dictionary containing the filters that were applied at each step and the resulting df.
    """
    prev_func = 'Raw'
    log_dict = {prev_func: df}
    for func_name, func_params in filter_process:
        # For just one string parameter (so that we don't iterate over the characters!)
        if type(func_params) == str:
            called = f'{func_name.__name__}(df, \'{func_params}\')'
            log_dict[called] = func_name(log_dict[prev_func], func_params)
        # Now we have only the list-like parameter sets which must be unpacked with *
        elif hasattr(func_params, '__iter__'):
            called = f'{func_name.__name__}(df, *{func_params})'
            log_dict[called] = func_name(log_dict[prev_func], *func_params)
        # For any other single parameters, e.g. a float
        else:
            called = f'{func_name.__name__}(df, {func_params})'
            log_dict[called] = func_name(log_dict[prev_func], func_params)
        if mode == 'individual':
            # Apply each function to the whole (initial) df
            prev_func = 'Raw'
        elif mode == 'cumulative':
            # Apply each function to the filtered df (the previous step)
            prev_func = called
        else:
            raise NameError('Not recognised')
    return log_dict

def filter_and_log_index(df, filter_process, mode='cumulative'):
    """
    Takes in a dataframe, and a filter_process - a list of lists.
    The filter_process should have the names of functions to sequentially apply
    to the dataframe as the first entry, and the parameters for each function as the second.
    Multiple parameters can be passed in as a tuple.
    At each step, the resulting dataframe's INDEX is pushed to an output dictionary,
    i.e. THE ROW NUMBERS FROM THE VALID ROWS.
    This is to preserve memory (one df that we reindex rather than several dfs with the same data)
    The key for the step is the string representation of the function that was called,
    WITH ITS PARAMETERS!
    This means we can track exactly which steps are cutting the most data.
    We can also save a specific filtering process because all the steps were passed to this function.
    Parameters
    ----------
    df: The input dataframe to be filtered
    filter_process: A list containing a series of 2 element lists.
    The first entry is the NAME of the function
    The second is the PARAMETERS to pass to it as its values (in a tuple if necessary).
    mode: Cumulative means each filter is applied to the result of the previous step.
    Individual means each filter is applied to the raw df that was passed in initially.

    Returns
    -------
    A dictionary containing the filters that were applied at each step and the resulting index,
    i.e. the row numbers that passed the filter.
    """
    prev_func = 'Raw'
    log_dict = {prev_func: df.index}
    for func_name, func_params in filter_process:
        # For just one string parameter (so that we don't iterate over the characters!)
        if type(func_params) == str:
            called = f'{func_name.__name__}(df, \'{func_params}\')'
            log_dict[called] = func_name(df.loc[log_dict[prev_func]], func_params).index
        # Now we have only the list-like parameter sets which must be unpacked with *
        elif hasattr(func_params, '__iter__'):
            called = f'{func_name.__name__}(df, *{func_params})'
            log_dict[called] = func_name(df.loc[log_dict[prev_func]], *func_params).index
        # For any other single parameters, e.g. a float
        else:
            called = f'{func_name.__name__}(df, {func_params})'
            log_dict[called] = func_name(df.loc[log_dict[prev_func]], func_params).index
        if mode == 'individual':
            # Apply each function to the whole (initial) df
            prev_func = 'Raw'
        elif mode == 'cumulative':
            # Apply each function to the filtered df (the previous step)
            prev_func = called
        else:
            raise NameError('Not recognised')
    # This is just here so you don't have to remember what the last filter was
    # You can get to the last state with the 'Final' key
    log_dict['Final'] = log_dict[prev_func]
    return log_dict

if __name__ == '__main__':
    lhc_df = make_clean_df(lhc_data_fpath)
    sim_df = make_clean_df(sim_data_fpath)

    lhc_log = filter_and_log_index(lhc_df, filter_process=filters_and_params, mode='cumulative')
    sim_log = filter_and_log_index(sim_df, filter_process=filters_and_params, mode='cumulative')

    print('For the real data:')
    for key in lhc_log:
        print(f'{key}, {len(lhc_log[key])} rows remaining')

    print('For the simulated data:')
    for key in sim_log:
        print(f'{key}, {len(sim_log[key])} rows remaining')
