import json

# TODO: make change to normalize_volatility function; breaks currently if json doesn't exist
# TODO: Can make the equations for these better but will required harder math; can be added later
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
