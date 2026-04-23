import yfinance as yf
import pandas as pd

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
