# 📉 Dynamic PCA Engine 

## Overview
The **Dynamic PCA Engine** is a preprocessing module designed to reduce dimensional complexity in macroeconomic datasets by transforming highly correlated variables into a smaller set of uncorrelated principal components. It is used as a feature engineering step before multivariate regression and machine learning models.

## 🎯 Objective
The goal of the engine is to:
* **Identify** highly correlated macro variables.
* **Reduce** redundancy through dimensionality reduction.
* **Preserve** most of the dataset’s informational variance.
* **Improve** stability of downstream regression models.

## 🧠 Core Motivation
Macroeconomic datasets often contain variables that move together (e.g., inflation measures, interest rates, employment indicators). This creates **multicollinearity**, which leads to:
* Unstable regression coefficients.
* Inflated standard errors.
* Reduced interpretability of model outputs.

The PCA engine is introduced to eliminate this redundancy while retaining meaningful signal.

## 🧮 Core Concept
The engine applies **Principal Component Analysis (PCA)** to groups of highly correlated variables, transforming them into orthogonal components:

$$Z = XW$$

**Where:**
* $X$: Standardized input features.
* $W$: Eigenvector matrix (principal directions).
* $Z$: Transformed uncorrelated components.

Only enough components are retained to preserve a chosen level of variance (typically **90–95%**).

## ⚙️ How It Fixes Multicollinearity
Multicollinearity occurs when features are highly correlated, causing redundant information in regression models. The PCA engine resolves this by:

1.  **Grouping** variables with correlation above a specific threshold (e.g., **0.8**).
2.  **Converting** each group into orthogonal principal components.
3.  **Replacing** original correlated variables with these components in the feature set.

Because PCA outputs are **linearly uncorrelated** by construction, it eliminates multicollinearity within each group while preserving most of the original variance.