from .data_fetcher import DataFetcher
from .risk_metrics import RiskMetrics
from .normalize_metrics import NormalizeRiskMetrics
from .risk_model import SectorRiskModel
from .cache_manager import CacheManager

import pandas as pd
from pathlib import Path

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
            
        return sector_risk_data[etf_ticker], sector_data
        
    else:
        print("Pulling Existing Data")
        sector_data = cache_manager.load_data() # return existing data
    
    return sector_data[etf_ticker], sector_data
