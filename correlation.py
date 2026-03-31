import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from data_cleanse import * 
import os


def correlation(pd, ticker):
    '''
    Finds correlation between specific ETF and macro data
    '''
    corr_matrix = pd.select_dtypes(include='number').corr()

    print(corr_matrix)

    plt.figure(figsize=(8,8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='Greens')
    plt.title(f'Correlation Matrix of Macroeconomic Variables and {ticker}')
    plt.savefig(f'plots/{ticker}.png')
    plt.show()


def graph(MACRO, ETF,  ETF_name, MACRO_name):
    '''
        MACRO- the macro df, typically from master_macro_table.csv
        ETF- ETF df
        ETF_name- string you want displayed, ticker will do
        MACRO_name- string you want displayed for macro measurement

        problems: units, not every macro is the same
    '''
    # Put into one table
    data = pd.concat([ETF, MACRO], axis=1)
    data.columns = [f'{ETF_name}', f'{MACRO_name}']

    # visualize
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.plot(data.index, data[f'{ETF_name}'], color='tab:blue', label=f'{ETF_name}')
    ax1.set_ylabel(f'{ETF_name} Price', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.plot(data.index, data[f'{MACRO_name}'], color='tab:red', label=f'{MACRO_name}')
    ax2.set_ylabel(f'{MACRO_name} Price', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title(f'Quarterly Closing Prices: {ETF_name} vs {MACRO_name}')
    fig.tight_layout()
    plt.savefig(f'plots/{ETF_name}_vs_{MACRO_name}.png')
    plt.show()


def build_etf_macro_correlation(etf_dir="data/raw_data/ETFs", macro_dir="data/raw_data", save_path="plots/correlation_matrix.png"):
    """
    Builds a correlation matrix of all ETFs and macroeconomic data.
    """

    # ---- Load all macro files ----
    macro_files = [f for f in os.listdir(macro_dir) if f.endswith(".csv") and "ETFs" not in f]
    macro_dfs = []
    for file in macro_files:
        df = fix_pd(os.path.join(macro_dir, file))
        df = df.select_dtypes(include="number")
        
        # Fix naming: only append file name if multiple columns exist
        if len(df.columns) > 1:
            df.columns = [f"{file.replace('.csv','')}_{col}" for col in df.columns]
        else:
            df.columns = [df.columns[0]]  # keep original column name
        
        macro_dfs.append(df)
    
    macro_df = pd.concat(macro_dfs, axis=1)

    # ---- Load all ETF files ----
    etf_files = [f for f in os.listdir(etf_dir) if f.endswith(".csv")]
    etf_dfs = []
    for file in etf_files:
        df = fix_pd(os.path.join(etf_dir, file))
        if "Close" in df.columns:
            df = df[["Close"]]
        df.columns = [file.replace(".csv", "")]
        etf_dfs.append(df)

    etf_df = pd.concat(etf_dfs, axis=1)

    # ---- Merge ETFs + Macro ----
    combined_df = pd.concat([etf_df, macro_df], axis=1)
    combined_df = combined_df.dropna(how="any")

    # ---- Correlation ----
    corr_matrix = combined_df.corr()

    # ---- Plot ----
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_matrix, annot=False, cmap="RdYlGn", center=0)
    plt.title("Correlation Matrix: ETFs vs Macros")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()

    return corr_matrix

if __name__ == "__main__":
    corr_matrix = build_etf_macro_correlation()