# ⏳ Correlation Engine

## Overview
The **Correlation Engine** is a statistical modeling pipeline designed to uncover lead-lag relationships between macroeconomic indicators and sector ETFs. It systematically identifies whether changes in macro variables precede movements in ETF returns and determines the optimal lag structure governing these relationships.

This engine serves as the **Feature Discovery Layer**, enabling downstream machine learning models to incorporate time-aware macro signals for improved forecasting.

## 🎯 Objective
* **Detect Temporal Relationships:** Identify lead-lag interactions between macro variables and ETF returns.
* **Identify Optimal Lags:** Determine the time delay ($L$) at which a macro variable has the highest predictive power.
* **Evaluate Stability:** Measure the consistency of these relationships across multiple market regimes.
* **Output Configuration:** Generate JSON-based lag maps for automated feature engineering.

## 🧠 Core Methodology

### 1. Stationarity Enforcement
To avoid spurious correlations, the engine enforces stationarity using the **Augmented Dickey-Fuller (ADF) Test**.
* **Null Hypothesis ($H_0$):** The series has a unit root (is non-stationary).
* **Significance Level ($\alpha$):** 0.05.

**Transformation Pipeline:**
* **ETFs:** 
    1. Test Raw Price. 
    2. If $p \geq 0.05$, convert to **Returns**: $R_t = \frac{P_t - P_{t-1}}{P_{t-1}}$
    3. If still non-stationary, apply **First Difference**: $R'_t = R_t - R_{t-1}$

* **Macro Variables:** 
    1. Test Raw Series. 
    2. If $p \geq 0.05$, apply **First Difference**: $X'_t = X_t - X_{t-1}$
    3. If still non-stationary, apply **Second Difference**: $X''_t = X'_t - X'_{t-1}$

### 2. Rolling Window Segmentation (`chunkify`)
To capture time-varying relationships, the data is divided into overlapping windows, stepped annually:
* **Window Size:** $N \times 12$ months (where $N$ is years, typically 3 or 5).
* **Step Size:** 12 months (1 year).
* **Purpose:** Prevents the model from assuming a static relationship over decades and captures structural economic shifts.

### 3. Lagged Correlation Analysis
For every window, the engine computes correlations between ETFs and lagged macro variables.
* **Lag Definition ($L$):** Macro variable at $t-L$ is compared with the ETF at time $t$.
* **Correlation Search:** For $L \in [1, \mathrm{max\_lags}]$:
    $$\rho_{X,Y}(L) = \mathrm{Corr}(X_{t-L}, Y_t)$$

**Optimal Lag Selection ($L^*$):**
The engine identifies the lag that yields the highest absolute correlation:
$$L^* = \arg\max_L |\rho_{X,Y}(L)|$$

* **Threshold Gate:** Only correlations where $|\rho| \geq 0.30$ are considered "Valid Windows."

## ⚙️ System Architecture

### 🔹 `preprocessing.py`
Implements the recursive ADF test and transformation logic.
* **`isStationary(series)`**: Returns `True` if $p < 0.05$.
* **`enforce_stationary()`**: Iterates through columns, applying `pct_change()` or `diff()` and tracking the number of transformations performed.

### 🔹 `analyzer.py`
The computational core:
* **`chunkify()`**: Slices the master dataframe into temporal chunks.
* **`compute_lagged_correlations()`**: Performs the shifting and matrix-wise correlation comparison. Utilizes boolean masking to update the `best_lag_matrix`.
* **`aggregate_lags()`**: Uses `collections.Counter` to find the **Mode Lag** and calculates the **Stability Metric**:
$$\text{Stability} = \frac{\text{Frequency of Modal Lag}}{\text{Total Valid Windows}}$$

### 🔹 `engine.py` & `config_generator.py`
* **`run_correlation_engine()`**: Orchestrates the data flow from `master_df` to final metrics.
* **`generate_json_config()`**: Serializes findings for persistence.

## ⚖️ Key Design Tradeoffs
* **Absolute Correlation Maximization:** Detects both positive and negative relationships (e.g., rising Unemployment hurting retail vs. falling Interest Rates helping tech).
* **Modal Aggregation:** Prioritizes the **consistency** of a lead-lag relationship over its sheer magnitude, ensuring features are robust for ML training.
* **Vectorized Masking:** Updates the `best_corr_matrix` using `pd.DataFrame.where(~mask)`, avoiding inefficient nested loops.

## 📊 Feature Output Structure
The JSON output maps directly to the feature engineering step of the pipeline:
```json
{
    "XLK": {
        "CPI": {
            "lag": 2,
            "stability": 0.75,
            "valid_windows": 8
        },
        "GDP": {
            "lag": 3,
            "stability": 0.30,
            "valid_windows": 6
        }
    },
    "XLE": {
        "Unrate": {
            "lag": -2,
            "stability": 0.80,
            "valid_windows": 9
        }
    }
}
```
| Metric | Meaning |
| :--- | :--- |
| **Lag** | The time delay (months) before a macro shift impacts the ETF. |
| **Stability** | Frequency of the modal lag / Total valid windows (Consistency). |
| **Valid Windows** | Number of time periods where $|\rho| \geq 0.30$. |