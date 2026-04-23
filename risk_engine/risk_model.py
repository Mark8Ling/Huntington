
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
