from data_cleanse import *
from linearRegression import linear_regression
from PCA import dynamic_pca
from correlation_engine.engine import run_correlation_engine
from correlation_engine.correlation import correlation

def create_linear_model(PROCESSING, TABLE_CONFIG, etf, display=False):
    '''
    A wrapper for all the steps to create a linear model. Takes processing and table configas arguemnts
    and then computes a master table, optimal lag, PCA, and then linear regression. 

    PPROCESSING: dict for ways to process the raw CSVs. 
    TABLE_CONFIG: dict for how to read the raw CSVs, and what transformations to apply.
    
    '''
    MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
    ETF = fix_pd(etf)

    ETF = ETF.pct_change()

    m_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')
    m_table = m_table[:240]
    # 2000-2020, cut off before covid

    macros_for_corr = list(MACRO.columns)
    yearly_period, lags = 5, 12     # For window to lag relationships, look into SE (standard error of correlation coefficient). Tells you how much noise to expect

    y = m_table["Close"]

    run_correlation_engine(m_table, macros_for_corr, ["Close"], yearly_period, lags, generate_config=True)

    valid_lag = []
    m_table, valid_lag = apply_lag("optimal_lags.json", m_table, stability_threshold=0.50)
    print(f"Valid lags applied: {valid_lag}")

    m_table = m_table.drop(columns=["Close"])

    MACRO_pca= dynamic_pca(m_table, correlation_threshold=0.80, variance_explained=0.90)
    MACRO_pca.to_csv('pca_macros.csv')

    osl, anova = linear_regression(MACRO_pca, y, etf)
    return osl, anova, valid_lag


if __name__ == "__main__":
    PROCESSING = {
        "read" : read_csv_standard,
        "quarterly" : read_quarterly,
        "MoM" : MoM,
        "interpolate_monthly" : interpolate_monthly,
        "YoY" : YoY,
        "enforce_stationary" : enforce_stationary,
        "log_diff" : log_diff
    }

    TABLE_CONFIG = { 
        "GDP": { 
            "path": "data/raw_data/GDP.csv", 
            "pipeline": ["read", "interpolate_monthly", "log_diff"], 
            "shift": 0 }, 
        "MCOILWTICO": { 
            "path": "data/raw_data/MCOILWTICO.csv", 
            "pipeline": ["read", "log_diff"], 
            "shift": 0 }, 
        # "PCEPI": { 
        #     "path": "data/raw_data/PCEPI.csv", 
        #     "pipeline": ["read", "log_diff"], 
        #     "shift": 0 },
        # "UNRATE": { 
        #     "path": "data/raw_data/UNRATE.csv", 
        #     "pipeline": ["read", "log_diff"], 
        #     "shift": 0 },
        # "FEDFUNDS": { 
        #     "path": "data/raw_data/FEDFUNDS.csv", 
        #     "pipeline": ["read", "log_diff"], 
        #     "shift": 0 }   
        }
    

    etf = 'data/raw_data/ETFs/XLE_monthly.csv'
    print(create_linear_model(PROCESSING, TABLE_CONFIG, etf, display=False))