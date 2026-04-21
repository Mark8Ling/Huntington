# 📊 Regression Modeling Suite

## Overview
The **Regression Modeling Suite** is a multi-method econometric framework designed to model and forecast ETF movements using macroeconomic variables. It includes linear regression, Ordinary Least Squares (OLS), Recursive Least Squares (RLS), and Rolling Window Regression, allowing for both static and time-adaptive modeling approaches.

The system is built with a strict focus on out-of-sample evaluation and financial time-series realism.

## 🎯 Objective
The goal of the regression suite is to:
* **Model** relationships between macroeconomic drivers and ETF returns.
* **Evaluate** both static and time-varying regression behavior.
* **Improve** forecasting robustness through rolling and recursive estimation.
* **Provide** strong out-of-sample evaluation metrics (e.g., $R^2$, directional accuracy).

## 🧮 Regression Models

### 1. Linear OLS Regression
Standard multivariate baseline for static relationships:
$$y = \beta_0 + \beta_1 x_1 + \cdots + \beta_n x_n + \epsilon$$
* **Method:** 80/20 train/test split using `statsmodels.OLS`.
* **Metrics:** Out-of-sample $R^2$, directional accuracy, and ANOVA significance.

### 2. Recursive OLS (RLS)
A dynamic model where coefficients evolve as new data arrives:
$$\beta_t = f(\beta_{t-1}, x_t, y_t)$$
* **Advantage:** Captures structural changes and tracks the evolution of macro-influence over time.

### 3. Rolling Window OLS
Estimates regression over a fixed-size moving window to handle non-stationarity:
$$\beta_t = \text{OLS}(y_{t-w:t}, x_{t-w:t})$$
* **Implementation:** Utilizes `RollingOLS` to re-estimate the model at each time step.

## 🧠 Addressing the “Three Hitters”
Standard multivariate regression in financial time series faces three major challenges that lead to biased inference and unstable coefficients:

| Problem | Description |
| :--- | :--- |
| **Multicollinearity** | Macro variables are highly correlated, inflating the variance of coefficients. |
| **Heteroskedasticity** | Error variance is not constant over time, making OLS inefficient. |
| **Autocorrelation** | Residuals are correlated across time, violating the independence assumption. |

## ⚙️ Engineering Solutions

* **For Multicollinearity:** Integrates the **Dynamic PCA Engine** to reduce features into orthogonal components before regression.
* **For Heteroskedasticity:** Shifts focus to **out-of-sample evaluation** and directional accuracy rather than relying on potentially biased p-values.
* **For Autocorrelation:** Uses **Rolling/Recursive structures** to reduce long-memory dependency issues and ensure temporal separation.


## 📊 Evaluation Framework

### 🔹 Out-of-Sample $R^2$ ($R^2_{OOS}$)
Measures predictive performance on unseen data:
$$R^2_{OOS} = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}$$

### 🔹 Directional Accuracy
Evaluates if the model correctly predicts the "Up" vs. "Down" movement, which is often more critical in financial contexts than absolute magnitude.


## 📈 Model Variants Summary

| Model | Purpose | Strength |
| :--- | :--- | :--- |
| **Linear OLS** | Baseline static relationship | High interpretability |
| **Recursive OLS** | Adaptive coefficient learning | Captures regime shifts |
| **Rolling OLS** | Local time-window estimation | Handles non-stationarity |

## ⚙️ System Design
* **`linear_regression()`**: Standard OLS with ANOVA and train/test evaluation.
* **`recursive_ordinary_least_squares()`**: Sequential updates to track parameter drift.
* **`window_ordinary_least_squares()`**: Rolling estimation for non-stationary environments.
* **`Evaluation Layer`**: Computes directional accuracy and Confusion Matrix breakdowns.

## 🚀 Key Insight
This suite is a **time-adaptive econometric system**. By combining PCA for feature stability with recursive and rolling estimation, the framework remains robust against the inherent instability and shifting regimes of global macroeconomic data.