import pandas as pd
import numpy as np
import math

class RiskMetrics:
    def __init__(self, ticker: str, etf_prices: pd.DataFrame, sp500_prices: pd.DataFrame, top_holdings: pd.DataFrame, top_holdings_prices: pd.DataFrame):
        self.ticker = ticker
        self.etf_prices = etf_prices
        self.sp500_prices = sp500_prices
        self.top_holdings = top_holdings
        self.top_holdings_prices = top_holdings_prices

    def compute_volatility(self):
        """ Computes the standard deviation from daily etf returns """

        # convert close etf prices to daily returns 
        etf_returns = self.etf_prices[self.ticker].pct_change()
        etf_returns = etf_returns.dropna()

        # compute standard deviation (daily volatility) of returns; convert to annualized
        daily_volatility = etf_returns.std()
        annualized_volatility = daily_volatility * math.sqrt(252)

        return annualized_volatility

    def compute_beta(self):
        """ 
            Compares how the etf moves in comparison to the S&P 500 
                - Baseline beta of 1.0
                - If market beta > 1, than etf is more sensitive to sp500 movements
                - If market beta < 1, then etf is less sensitive to sp500 movements
            Market Beta Formula:
                - covariance(etf, sp500) / variance(sp500)
        """
        
        # combine etf and sp500 into one df
        etf_and_sp500 = self.etf_prices.join(self.sp500_prices, how='inner')

        # convert etf and sp500 prices to returns
        etf_and_sp500 = etf_and_sp500.pct_change()
        etf_and_sp500 = etf_and_sp500.dropna()

        # get covariance and variance
        sp500_ticker = '^GSPC'
        etf_sp500_cov = etf_and_sp500[self.ticker].cov(etf_and_sp500[sp500_ticker])
        sp500_var = etf_and_sp500[sp500_ticker].var()

        # compute market beta
        beta = etf_sp500_cov / sp500_var

        return beta
    
    def compute_holdings_correlation(self):
        """ 
            Given top 5-10 stocks in a sector, compute their pairwise correlations 
                - (+1 = Perfectly Positive): As one variable increases, the other increases proportionally.
                - (0 = No Correlation): No linear relationship exists between the variables.
                - (-1 = Perfectly Negative): As one variable increases, the other decreases proportionally.
        """
        # enforce a column order that everything must adhere too
        ticker_order = self.top_holdings["Ticker"].tolist()
        
        # convert close etf prices to daily returns 
        top_holdings_returns = self.top_holdings_prices.pct_change()
        top_holdings_returns = top_holdings_returns[ticker_order] # enforce column order
        top_holdings_returns = top_holdings_returns.dropna()

        # create correlation matrix
        corr_matrix = top_holdings_returns.corr()

        # create weights matrix 
        aligned_weights_vector = (
            self.top_holdings.set_index("Ticker")
            .loc[ticker_order, "Weight"]
            .values.reshape(-1, 1) / 100
        )
        weights_matrix = aligned_weights_vector @ aligned_weights_vector.T 

        # apply weights to correlation matrix
        weighted_corrs = corr_matrix * weights_matrix

        # create boolean matrix
        tickers = weighted_corrs.index  # assumes square matrix with same index/columns
        n = len(tickers)

        row_idx = np.arange(n).reshape(-1, 1) # row index grids
        col_idx = np.arange(n).reshape(1, -1) # col index grids

        bool_matrix = row_idx < col_idx # i < j mask (only retrieve upper triangle of matrix, no diagonal)
        
        # get weighted correlation average 
        numerator = weighted_corrs.where(bool_matrix).sum().sum()
        denominator = weights_matrix[bool_matrix].sum()
        sector_corr_score = numerator / denominator

        return sector_corr_score
