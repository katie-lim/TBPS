"""
This script will be the filtering process for a dataset.
This is to ensure that the same steps are applied to the simulated data as to the real data.

The filter_list contains a list of the NAMES functions to be applied to the dataset in order.
This is so that different filters can be turned on and off easily to aid with debugging and checks.

The param_list contains all of the parameters to be passed to each function, IN THE SAME ORDER.
If a function takes multiple parameters, they should be passed as a tuple.
"""
import pandas as pd
from data_tools_and_filters import *

lhc_data_fpath = 'data/total_dataset.pkl'
sim_data_fpath = 'data/acceptance_mc.pkl'

lhc_df = make_clean_df(lhc_data_fpath)
sim_df = make_clean_df(sim_data_fpath)

filters = [type_combined_filter, type_combined_filter, type_combined_filter, type_combined_filter]
params = ['mu_plus', 'mu_minus', ('K', 0.25), ('Pi', 0.3)]

def filter_and_log(df, filter_list=filters, param_list=params, mode='individual'):
    prev_func = 'Raw'
    log_dict = {prev_func: df}
    for func_name, func_params in zip(filter_list, param_list):
        if type(func_params) == str:
            called = f'{func_name.__name__}(df, \'{func_params}\')'
            log_dict[called] = func_name(log_dict[prev_func], func_params)
        elif hasattr(func_params, '__iter__'):
            called = f'{func_name.__name__}(df, *{func_params})'
            log_dict[called] = func_name(log_dict[prev_func], *func_params)
        else:
            called = f'{func_name}(df, {func_params})'
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

lhc_log = filter_and_log(lhc_df)
sim_log = filter_and_log(sim_df)

for key in lhc_log:
    print(key, len(lhc_log[key]))