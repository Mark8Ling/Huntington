import pandas as pd
import yfinance as yf
import json
import numpy as np
from statsmodels.tsa.stattools import adfuller


'''
Clean up CSV files and put them into a master set
'''
def read_csv_standard(csv_file):
    # Reads CSV file, making observation_date the index
    df = pd.read_csv(csv_file)
    
    if 'observation_date' not in df.columns:
        return "no observation_date found"
    
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df = df.sort_values('observation_date')
    df.set_index('observation_date', inplace=True)

    return df
        
def read_quarterly(df):
    # Takes df and extrapolates to quarterly average
    return df.resample('QE').mean()

def interpolate_monthly(df):
    # Takes quarterly measured data and interpolates it to monthly frequency, keeping a complete monthly date index.
    full_monthly_index = pd.date_range(
        start=df.index.min(),
        end=df.index.max(),
        freq='MS'
    )
    # Reindex to full monthly dates
    df = df.reindex(full_monthly_index)
    # Interpolate missing values
    df_monthly = df.interpolate(method='linear')
    # Restore index name
    df_monthly.index.name = 'observation_date'
    return df_monthly

def MoM(df):
    # Calculate month over month % change
    df = df.pct_change()
    df = df.dropna() # first row will be NaN
    return df

def diff(df):
    df = df.diff()
    df = df.dropna()
    return df

def YoY(df):
    """
    Takes data frame and finds % change from last year
    """
    numeric_cols = df.select_dtypes(include="number").columns
    if len(numeric_cols) != 1:
        raise ValueError(f"Expected exactly one numeric column, found {list(numeric_cols)}")
    
    df_yoy = df.pct_change(periods=12) * 100
    df_yoy = df_yoy.dropna()
    
    return df_yoy

def log_diff(df):
    return np.log(df).diff().dropna()

def fix_pd(csv_file):
    '''
    When you call a raw CSV file an index needs to be set
    
    :param csv_file: Description
    '''
    ETF = pd.read_csv(csv_file)
    ETF['observation_date'] = pd.to_datetime(ETF['observation_date'])
    ETF.set_index('observation_date', inplace=True)
    return ETF

def master_table(table_config, processing, name):
    dfs = []

    for series_name, cfg in table_config.items():
        df = None

        # Apply pipeline
        for step in cfg["pipeline"]:
            if df is None:
                df = processing[step](cfg["path"])
            else:
                df = processing[step](df)

        numeric_cols = df.select_dtypes(include="number").columns
        if len(numeric_cols) != 1:
            raise ValueError(
                f"{series_name} must have exactly one numeric data column, "
                f"found {list(numeric_cols)}"
            )

        df = df[numeric_cols]
        df.columns = [series_name]
        # Shift
        shift = cfg.get("shift", 0)
        if shift != 0:
            df[series_name] = df[series_name].shift(shift)

        dfs.append(df)

    # Merge all series on index
    master = dfs[0]
    for df in dfs[1:]:
        master = master.merge(df, left_index=True, right_index=True, how="inner")

    # Drop rows with NaNs created by shifts
    master = master.dropna()

    # Save
    master.to_csv(f"{name}.csv")

    return master

def get_ticker(ticker):
    '''
    Use yahoo finance API to get specific stock/ETF data
    '''
    # Cut off for master_table is 2025-07-01 
    df = yf.download(ticker, start="2000-01-01", end="2025-08-01",interval="1mo")

    df.reset_index(inplace=True)
    df.columns = ['observation_date', 'Close', 'High', 'Low', 'Open', 'Volume']
    df['observation_date'] = pd.to_datetime(df['observation_date'], errors='coerce')
    df = df.dropna(subset=['observation_date'])
    
    df.to_csv(f'{ticker}_monthly.csv')
    return df

def enforce_stationary(master_df: pd.DataFrame):
#    Slightly altered and not as mobile as the enfore_stationary in correlation_engine
    etf_columns = ["Close"] if "Close" in master_df.columns else []
    macro_columns = [col for col in master_df.columns if col not in etf_columns]

    master_df_transformed = master_df.copy() # just incase we want to use both df's in the future
    etf_transformations = {
       etf: [0, False] for etf in etf_columns # assume non-stationary (False)
       }
    macro_transformations = {
       macro: [0, False] for macro in macro_columns 
    }

    # ensure all etf columns are stationary
    for etf in etf_columns:      
       curr_etf = master_df[etf]
      
       if isStationary(curr_etf): # true == stationary
           etf_transformations[etf][1] = True
           continue
       else: 
          curr_etf = curr_etf.pct_change() # convert from etf monthly price levels to etf returns
          etf_transformations[etf][0] += 1

          if isStationary(curr_etf):
             etf_transformations[etf][1] = True
             master_df_transformed[etf] = curr_etf
             continue
          else:
             curr_etf = curr_etf.diff() 
             etf_transformations[etf][0] += 1
           
             if isStationary(curr_etf):
                etf_transformations[etf][1] = True  
            
             master_df_transformed[etf] = curr_etf
   
   # ensure all macro columns are stationary
    for macro in macro_columns:      
       curr_macro = master_df[macro]
      
       if isStationary(curr_macro): 
          macro_transformations[macro][1] = True
          continue
       else: 
          curr_macro = curr_macro.diff() 
          macro_transformations[macro][0] += 1

          if isStationary(curr_macro):
             macro_transformations[macro][1] = True
             master_df_transformed[macro] = curr_macro
             continue
          else:
           curr_macro = curr_macro.diff() 
           macro_transformations[macro][0] += 1
           
           if isStationary(curr_macro):
               macro_transformations[macro][1] = True

           master_df_transformed[macro] = curr_macro
   
    return master_df_transformed

# Helper function
def isStationary(series: pd.Series) -> bool:
   '''
   Determine whether a time series is stationary using the 
   Augmented Dickey-Fuller (ADF) test.

   The ADF test evaluates the null hypothesis that a unit root
   is present in the series (i.e., the series is non-stationary).
   If the p-value is below the chosen significance level (typically 0.05),
   the null hypothesis is rejected and the series is considered stationary.
   
   :param series: Description
   :type series: pd.Series
   :return: Description
   :rtype: bool
   '''
   adf_test_results = adfuller(series.dropna()) # Returns a list of info about that pandas series
   p_value = adf_test_results[1] 
   significance_level = 0.05 # 95th percentile
   
   return p_value < significance_level # If true then data is stationary

def apply_lag(json_file, master_table, stability_threshold=0.75):
    with open(json_file, "r") as f:
        optimal_lags = json.load(f)

    # print(optimal_lags)
    target = "Close"
    lags_for_target = optimal_lags[target]

    # print(f"Optimal lags for {target}: {lags_for_target}")

    valid_lag = []
    for col, lag_info in lags_for_target.items():
        if col in master_table.columns:
            lag = lag_info.get("lag")
            stability = lag_info.get("stability", 0)
            if lag is not None and stability >= stability_threshold:
                master_table[col] = master_table[col].shift(lag)
                valid_lag.append((col, lag, stability))
    master_table = master_table.dropna()
    return master_table, valid_lag

if __name__ == "__main__":
    get_ticker("XLF")




