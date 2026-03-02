import pandas as pd
from .preprocessing import enforce_stationary
from .analyzer import chunkify, compute_lagged_correlations, aggregate_lags
from .config_generator import generate_json_config

def run_correlation_engine(master_df: pd.DataFrame, macro_columns: list, etf_columns: list, window_size: int, lags: int, generate_config=False):
    # ensure all data is stationary; transformations are 
    stationary_df, macro_transformations, etf_transformations = enforce_stationary(master_df, macro_columns, etf_columns)
    
    # create window chunks
    chunked_dfs = chunkify(stationary_df, window_size)

    # compute the best lag of each macro against each etf for each window
    all_window_lags = compute_lagged_correlations(chunked_dfs, macro_columns, etf_columns, lags)

    # determine the mode lag (best_lag) of each macro
    optimal_lags = aggregate_lags(all_window_lags)
    # print(f"Optimal Lags: {optimal_lags}")

    # only if the user wants to generate a json config 
    if generate_config:
        generate_json_config(optimal_lags, file_name='optimal_lags')
        generate_json_config(macro_transformations, file_name='macro_transformations')
        generate_json_config(etf_transformations, file_name='etf_transformations')

    return optimal_lags
