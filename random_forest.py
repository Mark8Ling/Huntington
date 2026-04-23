import os
import pandas as pd
from fredapi import Fred
from dotenv import load_dotenv

class FeatureEngineer():
    MACRO_PATH = "data/raw_data/macros"
    ETF_PATH = "data/raw_data/ETFs"
    
    All_MACROS = {
            # Growth
            "GDP": "GDP",
            "Industrial_Production": "INDPRO",
            "Retail_Sales": "RSAFS",
            "PMI_Proxy": "NAPM",  # ISM PMI

            # Inflation
            "CPI": "CPIAUCSL",
            "Core_CPI": "CPILFESL",
            "PPI": "PPIACO",

            # Rates
            "Fed_Funds_Rate": "FEDFUNDS",
            "10Y_Treasury": "GS10",
            "2Y_Treasury": "GS2",

            # Labor
            "Unemployment": "UNRATE",
            "Nonfarm_Payrolls": "PAYEMS",

            # Liquidity
            "M2": "M2SL",
            "Financial_Conditions": "NFCI",  # Chicago Fed

            # Consumer
            "Consumer_Confidence": "UMCSENT",
            "PCE": "PCE",

            # Commodities (FRED versions)
            "Oil_WTI": "DCOILWTICO",
            "Copper": "PCOPPUSDM"
        }
    
    def __init__(self):
        load_dotenv()
        self.fred = Fred(os.getenv("FRED_API_KEY"))
        os.makedirs(self.MACRO_PATH, exist_ok=True)

    def api_key(self):
        return self.fred
    def load_data(self, etf_ticker):
        macro_dfs = []

        # load or fetch macro data
        for name, macro_ticker in self.All_MACROS.items():
            file_path = os.path.join(self.MACRO_PATH, f"{name}.csv")

            if os.path.exists(file_path):
                print("file path exists")
                df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
            else:
                print("file path does not exist")
                data = self.fred.get_series(macro_ticker)

                df = pd.DataFrame(data, columns=[name])
                df.index.name = "Date"

                # Normalize to monthly frequency
                df = df.resample("ME").last()

                df.to_csv(file_path)

            macro_dfs.append(df)

        # merge the macro data
        print("merging all macros together")
        macro_df = pd.concat(macro_dfs, axis=1).sort_index()

        # load the etf
        print("loading etf file")
        etf_file = os.path.join(
            self.ETF_PATH, f"{etf_ticker.upper()}_monthly.csv"
        )

        if not os.path.exists(etf_file):
            raise FileNotFoundError(f"{etf_file} not found")

        etf_df = pd.read_csv(etf_file, parse_dates=["Date"], index_col="Date")

        etf_df = etf_df[["Close"]].rename(
            columns={"Close": etf_ticker.upper()}
        )

        # merge etf + macros
        df = pd.concat([etf_df, macro_df], axis=1).sort_index()

        # Fill lower-frequency macro gaps (GDP, etc.)
        df = df.ffill()

        # Drop remaining NaNs
        df = df.dropna()

        # Add derived features
        df["Yield_Spread"] = df["10Y_Treasury"] - df["2Y_Treasury"]

        return df

    def apply_lags():
        pass

    def create_target():
        pass

    def build_dataset():
        pass


class RandomForestModel():
    def __init():
        pass

    def train():
        pass

    def evaluate():
        pass

    def predict():
        pass

    def feature_importance_gini():
        pass

    def feature_importance_permutation():
        pass

class ScenarioEngine():
    def __init__():
        pass
    
    def run_base_case():
        pass

    def run_predefined_scenarios():
        pass

    def run_custom_scenario():
        pass


import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.inspection import permutation_importance
from sklearn.tree import plot_tree

# Takes in a pd with 1 ETF and all Macros; Will test with and without lags
def run_random_forest(
        df: pd.DataFrame, 
        n_estimators: int = 100, 
        max_depth: int = None, 
        min_sample_splits: int = 2, 
        max_features: str | int | None = "sqrt"
):
    """ Allows for hyper parameters, if not then a default forest will develop """

    metrics = [] # For storing output: will prolly need to change when later adding to frontend
    regressors = [] # Stores all regressors incase we want to do something with all of them like check regime changes
    
    # Slice df to separate etf and macros
    X = df.iloc[:, 1:] # all macro columns
    y = df.iloc[:, 0] # etf should be first column of df TODO make into etf returns

    # Split the data (ex: 2000-2004 --> 2000-2005 --> 2000-2006)
    tscv = TimeSeriesSplit(n_splits=15)

    # Loop through each time split
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]  
        
        # Train the model
        regressor = _train_model(X_train, y_train, n_estimators, max_depth, min_sample_splits, max_features)

        # Evaluate the model
        oob_score, mse, r2 = _evaluate_model(regressor, X_test, y_test)

        # Store output data
        regressors.append(regressor)
        metrics.append({
            "train_start": f'{X_train.index[0]}',
            "train_end": f'{X_train.index[-1]}',
            "test_start": f'{X_test.index[0]}',
            "test_end": f'{X_test.index[-1]}',
            "oob_score": oob_score,
            "mse": mse,
            "r2": r2
        })

        # Aggregate data
    
    return(
        regressors[-1], # Returns the last model that was train on the most amount of data
        X_test, # Needed for determining and visualizing feature importance
        y_test,
        metrics 
    ) 

def _train_model(X_train, y_train, n_estimators, max_depth, min_sample_splits, max_features):
    # Create regressor model
    regressor = RandomForestRegressor(
        n_estimators, # number of trees in the forest
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_sample_splits,
        max_features=max_features,
        oob_score=True
    )

    # Train the model
    regressor.fit(X_train, y_train)

    return regressor

# Should add more metrics like MAE, Adjusted R-squared, etc.
def _evaluate_model(regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series):
    y_pred = regressor.predict(X_test) # needed for determining mse and r2 values
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return regressor.oob_score_, mse, r2

def gini_feat_imp(regressor: RandomForestRegressor, features: list):
    # obtain default feature importances from running the model
    feature_importance = regressor.feature_importances_

    # sort features according to importance
    sorted_idx = np.argsort(feature_importance) # ranks the array positions by value
    pos = np.arange(sorted_idx.shape[0])

    # Visualize
    plt.barh(pos, feature_importance[sorted_idx], align="center")
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.title("Gini Feature Importance")
    plt.xlabel("Gini Importance")
    plt.ylabel("Macro-Indicators")
    plt.show()

def perm_feat_imp(regressor: RandomForestRegressor, X_test: pd.DataFrame, y_test: pd.Series, features: list):
    """
        How it works:
            1) Train the model first
            2) Measure baseline accuracy on the test set
            3) Shuffle one feature at a time across all test samples
            4) Measure new accuracy
            5) Feature Importance = baseline accuracy - shuffled accuracy  
    """
    results = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state=0, n_jobs=-1)

    # Sort importances
    sorted_idx = np.argsort(results.importances_mean)
    pos = np.arange(sorted_idx.shape[0])

    # Visualize 
    plt.barh(pos, results.importances_mean[sorted_idx], xerr=results.importances_std[sorted_idx], align='center')
    plt.yticks(pos, np.array(features)[sorted_idx])
    plt.xlim(left=0)
    plt.xlabel("Mean Decrease in Accuracy")
    plt.title("Permutation Feature Importance")
    plt.show()

def plot_individual_tree(regressor: RandomForestRegressor, features: list):
    """ Visualizes an individual tree in the forest, showing the decision making at each step """
   
    tree_to_plot = regressor.estimators_[0]
    plt.figure(figsize=(20,10))
    plot_tree(tree_to_plot, feature_names=features, filled=True, rounded=True, fontsize=10)
    plt.title("Decision Tree from Random Forest")
    plt.show()