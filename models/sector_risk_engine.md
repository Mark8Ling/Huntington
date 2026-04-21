# 📊 Sector Risk Engine

## Overview
The **Sector Risk Engine** is a quantitative model designed to evaluate the overall risk profile of a given GICS sector. It aggregates multiple dimensions of risk including volatility, market sensitivity, and internal holdings diversification into a single, interpretable risk score.

This model is part of a broader analytics platform that combines statistical modeling, machine learning, and financial theory to generate insights across equity sectors.

## 🎯 Objective
The goal of the Sector Risk Engine is to:
* **Quantify** sector-level risk using multiple financial metrics.
* **Normalize** different risk dimensions onto a comparable scale.
* **Aggregate** these metrics into a unified risk score.
* **Provide** a human-readable interpretation of the results.

## 🧠 Model Components
The model evaluates risk across three core dimensions:

### 1. Volatility
* Measures the variability of ETF returns.
* Computed as the standard deviation of daily returns.
* **Annualized using:**
$$
\sigma_{annual} = \sigma_{daily} \times \sqrt{252}
$$
* **Interpretation:** Higher volatility $\rightarrow$ larger price swings $\rightarrow$ higher risk.

### 2. Beta (Market Sensitivity)
* Measures how sensitive the ETF is to overall market movements (S&P 500).
* **Computed using:**
$$
\beta = \frac{\mathrm{Cov}(R_{ETF}, R_{S\&P500})}{\mathrm{Var}(R_{S\&P500})}
$$
* **Interpretation:** 
    * $\beta > 1$: Etf amplifies, and is more sensitive to market movements (Aggressive).
    * $\beta < 1$: Etf dampens, and is less sensitive to market movements (Defensive).

### 3. Holdings Correlation
* Measures how correlated the top holdings are with each other.
* Based on the top 10 holdings of the ETF (representing 70–99% of total value).
* Uses a **weighted pairwise correlation** approach to prioritize "Mega-Cap" influence.
* **Key Steps:**
    1. Convert price series $\rightarrow$ daily returns.
    2. Compute correlation matrix.
    3. Apply weights based on ETF allocation.
    4. Extract upper triangle (unique pairs only) using boolean masking.
    5. Compute weighted average correlation.
* **Interpretation:** High correlation $\rightarrow$ low diversification $\rightarrow$ higher risk.

## ⚖️ Normalization Strategy
To ensure comparability, each metric is transformed onto a $0-1$ scale:

* **Volatility (Min-Max Scaling):**
$$
\mathrm{norm\_vol} = \frac{vol - \min(vol)}{\max(vol) - \min(vol)}
$$

* **Beta (Distance from 1):**
$$
\mathrm{norm\_beta} = \min(|\beta - 1|, 1)
$$

* **Holdings Correlation:**
$$
\mathrm{norm\_corr} = \frac{corr + 1}{2}
$$

## 🧮 Risk Score Formula
The final sector risk score is computed as a weighted sum:

$$
\mathrm{Risk\ Score} = (0.5 \cdot \mathrm{norm\_vol}) + (0.3 \cdot \mathrm{norm\_beta}) + (0.2 \cdot \mathrm{norm\_corr})
$$

### Weighting Rationale:
* **Volatility (50%):** Primary driver of realized risk.
* **Beta (30%):** Captures external market exposure.
* **Correlation (20%):** Captures internal structural fragility.

## 📊 Output Example
The engine produces a JSON-based output for each sector:

```json
{
  "XLK": {
    "volatility": 0.258,
    "beta": 1.17,
    "holdings_correlation": 0.53,
    "normalized_volatility": 0.78,
    "normalized_beta": 0.17,
    "normalized_correlations": 0.76,
    "risk_score": 0.65,
    "last_updated": "2026-04-19"
  }
}
```

| Score Range | Risk Level |
| :--- | :--- |
| **< 0.30** | Low Risk |
| **0.30 – 0.60** | Moderate Risk |
| **0.60 – 0.80** | High Risk |
| **> 0.80** | Very High Risk |

## ⚙️ System Design
The engine follows a modular, object-oriented architecture:
* **`DataFetcher`**: Handles `yfinance` API calls and Excel holdings ingestion.
* **`RiskMetrics`**: Computes raw financial statistics.
* **`NormalizeRiskMetrics`**: Scales features for model input.
* **`SectorRiskModel`**: Aggregates metrics and generates NLP interpretations.
* **`CacheManager`**: Handles JSON persistence and data staleness logic (30-day rule).

## ⚖️ Key Design Tradeoffs
* **Relative Scaling**: Chose Min-Max scaling for volatility to enable cross-sector comparisons, though it requires a full sector dataset to be meaningful.
* **Beta Neutrality**: Used distance-from-one to capture risk as "deviation from the market," identifying both high-sensitivity and deep-defensive decoupling.
* **Matrix Optimization**: Utilized boolean masking (`row_idx < col_idx`) to avoid $O(n^2)$ nested loops for correlation extraction.
