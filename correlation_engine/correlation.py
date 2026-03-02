import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd 


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


    
'''
ETF = fix_pd('data/cleanedData/XLE_quarterly.csv')
ETF = ETF['Close']
MACRO = fix_pd('monthly_master_macro_table.csv')

master_table = MACRO.merge(ETF, on='observation_date', how='left')

print(master_table.head)

correlation(master_table, 'XLE')

MACRO_specific = MACRO['PCEPI']
graph(MACRO_specific, ETF, "XLE", "PCEPI")
'''

