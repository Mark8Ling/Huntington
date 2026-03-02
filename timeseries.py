import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt
from data_cleanse import *

'''
this is j ai slob... come back later
'''

# ===========================
# 1️⃣ Load and process data
# ===========================
etf = 'data/raw_data/ETFs/XLE_monthly.csv'
df = fix_pd(etf)

PROCESSING = {
    "read" : read_csv_standard,
    "quarterly" : read_quarterly,
    "MoM" : MoM,
    "interpolate_monthly" : interpolate_monthly,
    "YoY" : YoY,
    "enforce_stationary" : enforce_stationary,
    "log_diff" : log_diff
}

TABLE_CONFIG = { 
    "GDP": { 
        "path": "data/raw_data/GDP.csv", 
        "pipeline": ["read", "interpolate_monthly", "log_diff"], 
        "shift": 0 }, 
    "MCOILWTICO": { 
        "path": "data/raw_data/MCOILWTICO.csv", 
        "pipeline": ["read", "log_diff"], 
        "shift": 0 }, 
}

X = master_table(TABLE_CONFIG, PROCESSING, "all_macros")
y = df['Close'].pct_change().dropna()

# Make sure indices are aligned
y, X = y.align(X, join='inner')

# ===========================
# 2️⃣ Split into train/test
# ===========================
train_size = int(len(y) * 0.8)
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

print(f"Training size: {len(y_train)}, Test size: {len(y_test)}")

# ===========================
# 3️⃣ Fit ARIMAX model
# ===========================
p, d, q = 1, 1, 1
model = ARIMA(endog=y_train, exog=X_train, order=(p,d,q))
model_fit = model.fit()
print(model_fit.summary())

# ===========================
# 4️⃣ Forecast on test set
# ===========================
forecast = model_fit.get_forecast(steps=len(y_test), exog=X_test)
y_pred = forecast.predicted_mean

# Confidence intervals (optional)
forecast_ci = forecast.conf_int()

# ===========================
# 5️⃣ Evaluate performance
# ===========================
# Out-of-sample R²
ss_res = np.sum((y_test - y_pred)**2)
ss_tot = np.sum((y_test - y_test.mean())**2)
oos_r2 = 1 - ss_res/ss_tot
print(f"Out-of-sample R²: {oos_r2:.4f}")

# Compute month-over-month changes for test and predicted
y_test_diff = y_test.diff().dropna()
y_pred_diff = y_pred.diff().dropna()

# Compare the directions of the changes
direction_correct = np.sign(y_test_diff) == np.sign(y_pred_diff)
directional_accuracy = direction_correct.mean() * 100

print(f"Directional accuracy: {directional_accuracy:.2f}%")

# ===========================
# 6️⃣ Plot results
# ===========================
plt.figure(figsize=(12,6))
plt.plot(y_train.index, y_train, label='Train')
plt.plot(y_test.index, y_test, label='Actual Test')
plt.plot(y_test.index, y_pred, label='Forecast', color='orange')
plt.fill_between(forecast_ci.index, 
                 forecast_ci.iloc[:,0], 
                 forecast_ci.iloc[:,1], color='orange', alpha=0.2)
plt.title('ARIMAX ETF Forecast (Train/Test Split)')
plt.xlabel('Date')
plt.ylabel('ETF Returns')
plt.legend()
plt.show()