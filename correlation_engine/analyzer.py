import pandas as pd
import statistics

def chunkify(master_df: pd.DataFrame, yearly_periods: int) -> list:
    num_of_rows = yearly_periods * 12 # Get number of months
    length_of_df = len(master_df)
    all_chunks = []

    for i in range(0, length_of_df, 12): # Jump 1 year after each chunk
        start_index = i
        end_index = i + num_of_rows

        # grab a chunk
        chunk = master_df.iloc[start_index:end_index]

        # Only want to include equal length chunks
        # Ex: we use 3 years as our window, 25 / 3 will have a remainder which we don't want
        if len(chunk) == num_of_rows:
            all_chunks.append(chunk)

    return all_chunks

def compute_lagged_correlations(chunked_df: list[pd.DataFrame], macro_columns: list, etf_columns: list, num_of_lags: int) -> dict:
    start_lag = 1
    end_lag = num_of_lags + 1

    all_window_lags = {etf: {macro: [] for macro in macro_columns} for etf in etf_columns}

    for window in chunked_df: 
        temp_etf_df = window[etf_columns]
        temp_macro_df = window[macro_columns]

        best_corr_matrix = pd.DataFrame(0.0, index=macro_columns, columns=etf_columns)
        best_lag_matrix = pd.DataFrame(0, index=macro_columns, columns=etf_columns, dtype=int)

        for lag in range(start_lag, end_lag):  
            shifted_macro_df = temp_macro_df.shift(lag)
            combined = pd.concat([shifted_macro_df, temp_etf_df], axis=1)
            corr_matrix = combined.corr()
            curr_corr_matrix = corr_matrix.loc[macro_columns, etf_columns]
            # print(curr_corr_matrix)

            # Fixed this part
            mask = abs(curr_corr_matrix) > abs(best_corr_matrix)
            best_corr_matrix = best_corr_matrix.where(~mask, curr_corr_matrix)
            best_lag_matrix = best_lag_matrix.where(~mask, lag)
            # print(best_lag_matrix)
        
        threshold = 0.30  
        for etf in etf_columns:
            for macro in macro_columns:
                best_corr = best_corr_matrix.loc[macro, etf]
                best_lag = best_lag_matrix.loc[macro, etf]

                if abs(best_corr) >= threshold:
                    all_window_lags[etf][macro].append(int(best_lag))
                else: # no statistically significant correlation
                    all_window_lags[etf][macro].append(None)
    # print(all_window_lags)
    return all_window_lags

from collections import Counter

def aggregate_lags(lagged_correlations: dict) -> dict:
    results = {}
    for etf in lagged_correlations:
        results[etf] = {}
        for macro in lagged_correlations[etf]:
            lags = (lagged_correlations[etf][macro])
            # print(f"{macro}: {lags}")
            results[etf][macro] = {
                "lag": None,
                "stability": 0,
                "valid_windows": 0
            }
            not_none_lags = [l for l in lags if l is not None]
            if len(not_none_lags) == 0:
                continue
            else:
                counts = Counter(not_none_lags)
                num, freq = counts.most_common(1)[0]
                results[etf][macro] = {
                    "lag": num,
                    "stability": freq/len(not_none_lags),
                    "valid_windows": len(not_none_lags)
                }
                # print((f"{macro} - Modal Lag: {num}, Frequency: {freq}, Valid Windows: {results[etf][macro]['valid_windows']}"))
    return results