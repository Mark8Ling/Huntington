from data_cleanse import *
from linearRegression import linear_regression, recursive_ordinary_least_squares, window_ordinary_least_squares
from PCA import dynamic_pca
from correlation_engine.engine import run_correlation_engine
from correlation_engine.correlation import correlation
from statsmodels.graphics.tsaplots import plot_acf
import matplotlib.pyplot as plt


def create_linear_model(
        PROCESSING,
        TABLE_CONFIG,
        etf,
        use_lag=True,
        use_pca=True,
        corr_threshold=0.80,
        variance_explained=0.90,
        stability_threshold=0.50,
        display=False
    ):

    valid_lag = []

    MACRO = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
    ETF = fix_pd(etf)
    # print(ETF.head())
    ETF = ETF.pct_change()    
    
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # ETF = ETF[:240]
    # print(ETF["Close"])
    # print(ETF["Close"].describe())
    # print(ETF["Close"].autocorr(lag=12))
    # plot_acf(ETF["Close"], lags=12)
    # plt.show()
    
    m_table = MACRO.merge(ETF[['Close']], on='observation_date', how='left')
    m_table = m_table[:240]
    print(m_table)

    macros_for_corr = list(MACRO.columns)
    yearly_period, lags = 5, 12

    y = m_table["Close"]

    if use_lag:
        run_correlation_engine(
            m_table,
            macros_for_corr,
            ["Close"],
            yearly_period,
            lags,
            generate_config=True
        )

        m_table, valid_lag = apply_lag(
            "optimal_lags.json",
            m_table,
            stability_threshold=stability_threshold
        )

    print(valid_lag)
    # NOW remove Close (after lag engine is done)
    m_table = m_table.drop(columns=["Close"])

    if use_pca:
        MACRO_final = dynamic_pca(
            m_table,
            correlation_threshold=corr_threshold,
            variance_explained=variance_explained
        )
        MACRO_final.to_csv("pca_macros.csv")
    else:
        MACRO_final = m_table

    # print(MACRO_final.head())
    osl, anova = linear_regression(MACRO_final, y, etf)
    print("--------------------------------------------------********************-----------")
    summary, final_results= recursive_ordinary_least_squares(MACRO_final, y, etf, output_dir="reports/images")
    print(summary)
    print(final_results)


    # all_params_df, final_results = window_ordinary_least_squares(MACRO_final, y, etf, output_dir="reports/images")
    # print(all_params_df)
    # print(final_results)

    
    return summary, final_results, valid_lag
    # return osl, anova, valid_lag


if __name__ == "__main__":
    PROCESSING = {
        "read" : read_csv_standard,
        "quarterly" : read_quarterly,
        "MoM" : MoM,
        "interpolate_monthly" : interpolate_monthly,
        "YoY" : YoY,
        "enforce_stationary" : enforce_stationary,
        "log_diff" : log_diff,
        "diff" : diff
    }


    TABLE_CONFIG = {
        "GDP": {
            "path": "data/raw_data/GDP.csv",
            "pipeline": ["read", "interpolate_monthly", "log_diff"],
            "shift": 0
        },
        "MCOILWTICO": {
            "path": "data/raw_data/MCOILWTICO.csv",
            "pipeline": ["read", "log_diff"],
            "shift": 0
        },
        "UNRATE": {
            "path": "data/raw_data/UNRATE.csv",
            "pipeline": ["read", "diff"],
            "shift": 1
        },
     }

    
    etf = 'data/raw_data/ETFs/XLP_monthly.csv'

    create_linear_model(
        PROCESSING,
        TABLE_CONFIG,
        etf,
        use_lag=True,
        use_pca=True,
        corr_threshold=0.80,
        variance_explained=0.90,
        stability_threshold=0.50,
        display=True)
    # print(create_linear_model(PROCESSING, TABLE_CONFIG, etf, display=False))


