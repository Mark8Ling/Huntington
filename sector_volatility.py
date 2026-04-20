import yfinance as yf
import pandas as pd
import numpy as np

import math
import json

from pathlib import Path

class DataFetcher:
    def __init__(self, etf_ticker: str):
        self.etf_ticker = etf_ticker
        self.start_date = '2000-01-01'

    def _get_price_series(self, ticker):
        """ Helper function for pulling data from yfinance"""

        data = yf.download(ticker, start=self.start_date)

        # Clean and format data into a DataFrame
        data.reset_index(inplace=True)  
        data.columns = ['observation_date', 'Close', 'High', 'Low', 'Open', 'Volume']
        data['observation_date'] = pd.to_datetime(data['observation_date'], errors='coerce')
        data = data.dropna(subset=['observation_date'])
        data = data.set_index('observation_date')
        data = data.drop(columns=['High', 'Low', 'Open', 'Volume'])
        data.rename(columns={'Close': f'{ticker}'}, inplace=True)

        return data

    def get_etf_prices(self):
        """ pull etf prices from yfinance """
        
        etf_prices = self._get_price_series(self.etf_ticker)

        return etf_prices
    
    def get_sp500_prices(self):
        """ pull S&P500 prices from yfinance """
        
        sp500_ticker = '^GSPC'
        sp500_prices = self._get_price_series(sp500_ticker)

        return sp500_prices

    def get_holdings(self):  
        """ pull top holdings from yfinance """

        sector_info = pd.read_excel(rf'data\raw_data\ETFs\etf_holdings\{self.etf_ticker}.xlsx', sheet_name='holdings')
        holdings_info = sector_info[['Name', 'Ticker', 'Weight']].head(10) # pull the top 10 holdings in the sector

        # Create dict of company names and associated tickers for output
        # !!! EXTRA (Not necessary to model but can be used for visualizations)
        top_holdings = {}
        for index, row in holdings_info.iterrows():
             company = row['Name'] # get company name
             ticker = row['Ticker'] # get company ticker symbol
             top_holdings[company] = ticker # add to dict

        return holdings_info

    def get_holdings_prices(self, top_holdings):
        """ pull top holding prices from yfinance """
        
        holdings_tickers = top_holdings['Ticker'].to_list()
        raw_data = yf.download(holdings_tickers, start=self.start_date)
        raw_data = raw_data.drop(columns=['High', 'Low', 'Open', 'Volume'])

        # Columns are Multi-indexed by default; so each column is ['Close': 'ETF_Ticker']
        # flatten column names to only be the etf tickers
        close_prices = raw_data['Close']

        return close_prices

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

# TODO: make change to normalize_volatility function; breaks currently if json doesn't exist
class NormalizeRiskMetrics:
    def __init__(self, etf_ticker: str, volatility: int, beta: int, holdings_corr: int):
        self.etf_ticker = etf_ticker
        self.volatility = volatility
        self.beta = beta
        self.holdings_corr = holdings_corr

    def normalize_volatility(self):
        """ Normalize sector volatility using min-max scaling to convert raw values into a comparable 0-1 range across all sectors """
        
        with open('sector_risk_data.json', 'r') as f:
            sector_risk_data = json.load(f)
        
        # collect all valid volatility values
        vols = [
            sector_risk_data[etf]["volatility"]
            for etf in sector_risk_data
            if "volatility" in sector_risk_data[etf]
        ]

        min_vol = min(vols)
        max_vol = max(vols)

        # normalize each sector's volatility
        for ticker in sector_risk_data:
            if "volatility" not in sector_risk_data[ticker]:
                continue

            vol = sector_risk_data[ticker]["volatility"]

            # handle edge case where all vols are equal
            if max_vol == min_vol:
                vol_norm = 0.5
            else:
                vol_norm = (vol - min_vol) / (max_vol - min_vol)

            sector_risk_data[ticker]["normalized_volatility"] = vol_norm

        # pull norm_vol for this specific sector
        normalized_volatility = sector_risk_data[self.etf_ticker]["normalized_volatility"]
        
        return normalized_volatility

    def normalize_beta(self):
        return min(abs(self.beta - 1), 1) # determines distance from 1

    def normalize_holdings_corr(self):
        return (self.holdings_corr + 1) / 2

# TODO: make edits to interpret_risk_score function
class SectorRiskModel:
    def __init__(self, etf_ticker: str, norm_vol: int, norm_beta: int, norm_holdings_corr: int):
        self.etf_ticker = etf_ticker
        self.norm_vol = norm_vol
        self.norm_beta = norm_beta
        self.norm_holdings_corr = norm_holdings_corr
    
    def generate_sector_risk(self):
        """ Aggregate all three metrics into one universal sector risk score """

        # determine weights per metric; sum to 1
        vol_weight = 0.5
        beta_weight = 0.3
        corrs_weight = 0.2

        # compute sector risk score
        risk_score = (
            self.norm_vol * vol_weight +
            self.norm_beta * beta_weight +
            self.norm_holdings_corr * corrs_weight
        )
        
        return risk_score
    
    def interpret_risk_score(self, risk_score: int):
        """
            Generates an xplanation of the ETF risk profile based 
            on normalized risk metrics and the final risk score.
        """

        # interpret risk score
        if risk_score < 0.3:
            label = "Low Risk"
        elif risk_score < 0.6:
            label = "Moderate Risk"
        elif risk_score < 0.8:
            label = "High Risk"
        else:
            label = "Very High Risk"

        # interpret volatility
        if self.norm_vol < 0.3:
            vol_desc = "relatively stable compared to other sectors"
        elif self.norm_vol < 0.6:
            vol_desc = "shows moderate price fluctuations"
        else:
            vol_desc = "experiences high price volatility and sharp movements"

        # interpret beta
        if self.norm_beta > 1.1:
            beta_desc = "amplifies overall market movements"
        elif self.norm_beta < 0.9:
            beta_desc = "is less sensitive to overall market movements"
        else:
            beta_desc = "moves closely in line with the broader market"

        # interpret correlations
        if self.norm_holdings_corr < 0.3:
            corr_desc = "benefits from diversification among holdings"
        elif self.norm_holdings_corr < 0.7:
            corr_desc = "has moderate co-movement between holdings"
        else:
            corr_desc = "shows strong internal correlation, limiting diversification"

        # generate final interpretation
        interpretation = (
            f"Risk Level: {label} (Score: {risk_score:.2f})\n\n"
            f"The {self.etf_ticker} sector's risk profile is driven by a combination of volatility, "
            f"market sensitivity, and internal correlation among holdings.\n\n"
            f"- Volatility (raw: {self.norm_vol:.2f}, normalized: {self.norm_vol:.2f}): "
            f"The sector {vol_desc}.\n"
            f"- Beta (raw: {self.norm_beta:.2f}, normalized deviation: {self.norm_beta:.2f}): "
            f"The sector {beta_desc}.\n"
            f"- Holdings Correlation (raw: {self.norm_holdings_corr:.2f}, normalized: {self.norm_holdings_corr:.2f}): "
            f"The sector {corr_desc}.\n\n"
            f"Overall, the combination of these factors indicates that this sector "
            f"is classified as {label.lower()}."
        )

        return interpretation

class CacheManager:
    def __init__(self, ticker: str):
        self.ticker = ticker

    def load_data(self):
        """ if the json data exists and is not stale, then simply pull the data from the json """
        
        with open('sector_risk_data.json', 'r') as f:
            sector_data = json.load(f)
        
        return sector_data

    def save(self, filepath, sector_risk_data: dict):
        """ Save the output from the SectorRiskModel into a json """
        
        with open(filepath, 'w') as f:
            json.dump(sector_risk_data, f, indent=4)

    def is_stale(self):
        """ Stale means that the last time the sector volatility was computed is > ago """
        with open('sector_risk_data.json', 'r') as f:
            sector_data = json.load(f)
        
        # get current date
        current_date = pd.Timestamp.today().date()

        # pull latest date from the json dataset
        previous_date = pd.to_datetime(sector_data[self.ticker]['last_updated']).date()

        # compute difference
        return (current_date - previous_date).days >= 30
    
def run_risk_engine(etf_ticker):
    """ How this will interact with the frontend and call all other functions and classes """

    file_path = Path(__file__).resolve().parent / "sector_risk_data.json"
    cache_manager = CacheManager(ticker=etf_ticker)

    # check if file doesn't exists or if data is out-of-date
    if not file_path.is_file() or cache_manager.is_stale():        
    # prepare to run pipeline 
        print("Preparing Pipeline...")    
        sector_risk_data = {}
        gics_sectors = ["XLK", "XLV", "XLF", "XLY", "XLP", "XLE", "XLI", "XLB", "XLU", "XLRE", "XLC"]

        for ticker in gics_sectors:
            try:
                # fetch data
                print('fetching data')
                data_fetcher = DataFetcher(ticker)

                etf = data_fetcher.get_etf_prices()
                sp500 = data_fetcher.get_sp500_prices()
                top_holdings = data_fetcher.get_holdings()
                top_holdings_prices = data_fetcher.get_holdings_prices(top_holdings)

                # generate risk metrics
                print('generating risk metrics')
                risk_metrics = RiskMetrics(
                    ticker=ticker, 
                    etf_prices=etf, 
                    sp500_prices=sp500, 
                    top_holdings=top_holdings, 
                    top_holdings_prices=top_holdings_prices
                )

                volatility = risk_metrics.compute_volatility()
                beta = risk_metrics.compute_beta()
                holdings_corr = risk_metrics.compute_holdings_correlation()

                # Normalize raw risk metrics onto a common scale (0-1) so they can be meaningfully combined into a single risk score
                # This prevents any one metric from dominating due to differences in magnitude rather than true economic importance.
                print('normalizing risk metrics')
                normalized_metrics = NormalizeRiskMetrics(etf_ticker=ticker, volatility=volatility, beta=beta, holdings_corr=holdings_corr)

                normalized_volatility = normalized_metrics.normalize_volatility()
                normalized_beta = normalized_metrics.normalize_beta()
                normalized_correlations = normalized_metrics.normalize_holdings_corr()

                # generate risk score
                print('generating risk score')
                risk_engine = SectorRiskModel(
                    etf_ticker=ticker, 
                    norm_vol=normalized_volatility, 
                    norm_beta=normalized_beta, 
                    norm_holdings_corr=normalized_correlations
                )

                risk_score = risk_engine.generate_sector_risk()
                interpreted_risk_score = risk_engine.interpret_risk_score(risk_score=risk_score)
                
                # add data to our output
                print('adding data to output')
                sector_risk_data[ticker] = {
                        "volatility": volatility,
                        "beta": beta,
                        "holdings_correlation": holdings_corr,
                        "normalized_volatility": normalized_volatility,
                        "normalized_beta": normalized_beta,
                        "normalized_correlations": normalized_correlations,
                        "risk_score": risk_score,
                        "last_updated": str(pd.Timestamp.today().date())
                    }

            except Exception as e:
                # prevents one failure from killing the full batch
                sector_risk_data[ticker] = {
                    "error": str(e),
                    "last_updated": str(pd.Timestamp.today().date())
                }

        # store data into json       
        cache_manager.save(file_path, sector_risk_data)
            
        return sector_risk_data[etf_ticker]
        
    else:
        print("Pulling Existing Data")
        sector_data = cache_manager.load_data() # return existing data
    
    return sector_data[etf_ticker]
