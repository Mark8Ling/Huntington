import pandas as pd
from data_cleanse import read_csv_standard

'''
Objective: Test whether macro variables lead lag, or if lag leads the target. Check that it is
stable over time to avoid building a model that follows noise and one-off shocks
'''

def chunkify(df, period=60):
    '''
    Splits a DataFrame into chunks of a given time period (in months).
    
    df- master macro sheet
    period- how long the chunks are in months

    Returns:
        List[pd.DataFrame]: List of DataFrame chunks
    '''

    chunk = []
    current_chunk = []

    for count, (index, row) in enumerate(df.iterrows(), start = 1):
        current_chunk.append(row)
        if count % period == 0:
            chunk.append(pd.DataFrame(current_chunk))
            current_chunk = []

    if current_chunk:
        # append remaining period
        chunk.append(pd.DataFrame(current_chunk))

    return chunk
    

def lagged_correlation(df, target_col, max_lag=12):
    """
    Compute the lag that maximizes the correlation between each macro variable 
    in a DataFrame and a target variable.

    For each column in `df` (excluding `target_col`), the function tests lags 
    from `-max_lag` to `+max_lag` months. It identifies the lag that produces 
    the **highest correlation** with the target variable. 

    This is useful for identifying leading or lagging relationships between 
    macro variables and a target, and for constructing lagged features for 
    predictive models. When viewed in multiple chunks, we can idenify patterns, 
    or lack there of.

    Parameters
    ----------
    df : pd.DataFrame
        A pandas DataFrame containing the time series data of macro variables.

    target_col : str
        The name of the column in `df` to measure correlation against. 

    max_lag : int, default=12
        Maximum lag (in months or time units) to test in either direction. 
        Negative lag means the variable leads the target while positive lag means 
        it lags the target.

    Returns
    -------
    pd.DataFrame
        A DataFrame with the following columns:
        - variable: name of the macro variable
        - best_lag: the lag (negative or positive) that maximizes absolute correlation
        - max_corr: the correlation value at the best lag
        """


    results = []

    for col in df.columns:
        if col == target_col:
            continue

        best_corr = None
        best_lag = None

        for lag in range(-max_lag, max_lag + 1):
            shifted = df[col].shift(lag)
            corr_df = pd.concat([shifted, df[target_col]], axis=1).dropna()
            if len(corr_df) < 3:  # need at least 3 points
                continue

            corr = corr_df.iloc[:, 0].corr(corr_df.iloc[:, 1])

            if best_corr is None or abs(corr) > abs(best_corr):
                best_corr = corr
                best_lag = lag

        results.append({
            'variable': col,
            'best_lag': best_lag,
            'max_corr': best_corr
        })

    return pd.DataFrame(results)


if __name__ == "__main__":
    macros = read_csv_standard('monthly_master_macro_table.csv')
    target_col = 'GDP'  
    chunks = chunkify(macros, period=24) # 2 year periods

    # Compute lagged correlations for each chunk
    all_chunk_results = []
    for i, chunk in enumerate(chunks):
        chunk_result = lagged_correlation(chunk, target_col, max_lag=12)
        chunk_result['chunk'] = i + 1
        all_chunk_results.append(chunk_result)

    # Combine all results
    final_results = pd.concat(all_chunk_results, ignore_index=True)

    # print(final_results)
    # Get a specific macro, easier to visualze change
    oil_results = final_results[final_results['variable'] == 'PCEPI']
    print(oil_results)