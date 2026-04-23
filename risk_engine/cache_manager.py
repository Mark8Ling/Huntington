import pandas as pd
import json

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
    