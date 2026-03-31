import pathlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, confusion_matrix, classification_report
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from statsmodels.stats.anova import anova_lm
import os


    
def linear_regression(x, y, etf, output_dir="reports/images"):
    os.makedirs(output_dir, exist_ok=True)

    df = x.copy()
    df["y"] = y

    train_size = int(len(df) * 0.80)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    predictors = " + ".join(x.columns)
    formula = f"y ~ {predictors}"

    model = smf.ols(formula, data=train).fit()

    y_pred = model.predict(test)

    results, directional_accuracy, r2_oos = model_testing(x, y, model, test)
    
    graph(results, train, test, y_pred, etf, output_dir, directional_accuracy, r2_oos)

    print(model.summary())
    print(anova_lm(model, typ=1))

    return model.summary(), anova_lm(model, typ=1)


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

def recursive_ordinary_least_squares(x, y, etf, output_dir="reports/images"):
    """
    Performs recursive OLS (RLS) on training data and makes recursive predictions on test data.
    Produces plots of coefficient evolution and predicted vs actual values.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Prepare dataframe
    df = x.copy()
    df["y"] = y

    # Split into train/test
    train_size = int(len(df) * 0.80)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    predictors = " + ".join(x.columns)
    formula = f"y ~ {predictors}"

    # Initialize RecursiveLS on training data
    reg_rls = sm.RecursiveLS.from_formula(formula, data=train)
    model_rls = reg_rls.fit()
    
    # Store evolving coefficients during training
    coeffs_over_time = pd.DataFrame(model_rls.filtered_state.T, 
                                    columns=["Intercept"] + list(x.columns),
                                    index=train.index)

    # Recursive out-of-sample predictions
    y_pred = []
    coefs = model_rls.params.copy()  # start with final training coefficients

    # Add constant column for dot-product
    X_test = sm.add_constant(test[x.columns])

    for t in range(len(test)):
        x_t = X_test.iloc[t].values
        y_t_pred = x_t @ coefs
        y_pred.append(y_t_pred)

        # Update coefficients recursively using this new observation
        # Statsmodels RecursiveLS doesn't expose a simple step-update,
        # so we can append the new observation and refit a single-step RLS
        tmp_df = pd.concat([train, test.iloc[:t+1]], axis=0)
        reg_tmp = sm.RecursiveLS.from_formula(formula, data=tmp_df)
        model_tmp = reg_tmp.fit()
        coefs = model_tmp.params  # update coefficients

    # Convert predictions to series
    y_pred = pd.Series(y_pred, index=test.index)

    # Plot coefficient evolution + recursive predictions
    fig, axes = plt.subplots(len(x.columns)+1, 1, figsize=(10, 3*(len(x.columns)+1)), sharex=True)

    # Plot training coefficient evolution
    for i, col in enumerate(coeffs_over_time.columns):
        axes[i].plot(coeffs_over_time.index, coeffs_over_time[col], label=f"{col} (train)")
        axes[i].plot(test.index, [coefs[i]]*len(test), '--', label=f"{col} (test start)")
        axes[i].set_ylabel(col)
        axes[i].legend(loc='upper right')

    # # Plot predictions vs actual
    # axes[-1].plot(test.index, test["y"], label="Actual", color='black')
    # axes[-1].plot(test.index, y_pred, label="Predicted (RLS)", color='red')
    # axes[-1].set_ylabel("y")
    # axes[-1].set_xlabel("Time")
    # axes[-1].legend(loc='upper left')

    plt.suptitle(f"Recursive OLS - {etf}")
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    plt.show()
    plt.savefig(os.path.join(output_dir, f"RLS_{etf}.png"))
    plt.close(fig)

    return model_rls.summary(), y_pred

def window_ordinary_least_squares(x, y, etf, output_dir="reports/images", window=120):
    """
    Rolling/window OLS with out-of-sample prediction at each step using the most recent rolling coefficients.
    """
    os.makedirs(output_dir, exist_ok=True)

    df = x.copy()
    df["y"] = y

    train_size = int(len(df) * 0.80)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    predictors = " + ".join(x.columns)
    formula = f"y ~ {predictors}"

    # Combine train + test for rolling coefficient calculation
    full_data = pd.concat([train, test])

    # Fit rolling OLS on full data (rolling windows only valid where enough history exists)
    reg_window = RollingOLS.from_formula(formula, window=window, data=full_data)
    model_window = reg_window.fit()

    # Make rolling predictions for the test set
    y_pred = []
    for t_idx, idx in enumerate(test.index):
        # Grab the last available rolling coefficient before the test index
        coefs = model_window.params.loc[:idx].iloc[-1].values  # array of [const, beta1, beta2, ...]
        x_t = np.r_[1, test.loc[idx, x.columns].values]        # add constant
        y_t_pred = x_t @ coefs
        y_pred.append(y_t_pred)

    # Compute metrics
    results_df, directional_accuracy, r2_oos = rls_model_testing(test, y_pred)

    # Plot
    rls_graph(results_df, train, test, y_pred, etf_name=etf, output_dir=output_dir,
              directional_accuracy=directional_accuracy, r2_oos=r2_oos)

    return None, None

def rls_graph(results_df, train, test, y_pred, etf_name="ETF", output_dir="reports/images", directional_accuracy=None, r2_oos=None):
    """
    Plots RecursiveLS predictions:
        Left: Train + Test + Predicted
        Right: Out-of-sample evaluation (Actual vs Predicted)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(14,6))

    # ---- Left Plot: Full time series ----
    ax1 = plt.subplot(1,2,1)
    ax1.plot(train.index, train["y"], label="Train (Actual)", color="black")
    ax1.plot(test.index, test["y"], label="Test (Actual)", linewidth=2, color="orange")
    ax1.plot(test.index, y_pred, label="Test (Predicted)", linestyle="--", color="green")
    ax1.axvline(test.index[0], color="black", linestyle=":", label="Train/Test Split")
    ax1.set_title("Train/Test with Predictions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # ---- Right Plot: OOS evaluation ----
    ax2 = plt.subplot(1,2,2)
    ax2.plot(results_df.index, results_df["Actual"], label="Actual", linewidth=2, color="orange")
    ax2.plot(results_df.index, results_df["Predicted"], label="Predicted", linestyle="--", color="green")
    ax2.set_title("Out-of-Sample: Actual vs Predicted")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    # ---- Metrics box ----
    if directional_accuracy is not None and r2_oos is not None:
        metrics_text = f"R² (OOS): {r2_oos:.4f}\nDirectional Accuracy: {directional_accuracy:.2%}"
        ax2.text(
            0.02, 0.98,
            metrics_text,
            transform=ax2.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
        )

    plt.tight_layout()

    # filepath = os.path.join(output_dir, f"{etf_name}_rls_results.png")
    # plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close()

def rls_model_testing(test, y_pred):
    """
    Computes evaluation metrics for RecursiveLS predictions.
    Returns:
        results_df: DataFrame with Actual, Predicted, Error, Direction info
        directional_accuracy: % of correct direction predictions
        r2_oos: out-of-sample R²
    """
    results_df = test.copy()
    results_df = results_df.rename(columns={"y": "Actual"})
    results_df["Predicted"] = y_pred

    # ---- Errors ----
    results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
    results_df["Squared_Error"] = results_df["Error"] ** 2

    # Month-over-month changes
    results_df["Actual_Change"] = results_df["Actual"].diff()
    results_df["Predicted_Change"] = results_df["Predicted"].diff()
    results_df = results_df.iloc[1:]  # drop first NaN

    # ---- Directional correctness ----
    results_df["Actual_Direction"] = results_df["Actual_Change"] > 0
    results_df["Predicted_Direction"] = results_df["Predicted_Change"] > 0
    results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]
    results_df["Direction_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

    directional_accuracy = results_df["Correct_Direction"].mean()

    # ---- Out-of-sample R² ----
    mean_train = results_df["Actual"].mean()  # approximate baseline
    sse_model = results_df["Squared_Error"].sum()
    sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum()
    r2_oos = 1 - (sse_model / sse_mean)

    # Clean up for display
    results_df.drop(['Correct_Direction', 'Predicted_Direction', 'Squared_Error', 'Actual_Change', 'Actual_Direction'], axis=1, inplace=True)
    
    return results_df, directional_accuracy, r2_oos

def model_testing(x, y, model, test):
    test = test.copy()
    results = []

    for idx, row in test.iterrows():
        y_actual = row["y"]
        y_pred = model.predict(row.to_frame().T).iloc[0]

        results.append({
            "Month": idx,
            "Actual": y_actual,
            "Predicted": y_pred
        })

    results_df = pd.DataFrame(results)
    results_df.set_index("Month", inplace=True)

    # ---- Errors & directional correctness ----
    results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
    results_df["Squared_Error"] = results_df["Error"] ** 2

    # Month-over-month changes
    results_df["Actual_Change"] = results_df["Actual"].diff()
    results_df["Predicted_Change"] = results_df["Predicted"].diff()
    results_df = results_df.iloc[1:]

    results_df["Actual_Direction"] = results_df["Actual_Change"] > 0
    results_df["Predicted_Direction"] = results_df["Predicted_Change"] > 0
    results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]
    results_df["Direction_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

    directional_accuracy = results_df["Correct_Direction"].mean()

    # ---- Direction-aware L/M/H ----
    pos = results_df[results_df["Actual_Change"] > 0]["Actual_Change"]
    neg = results_df[results_df["Actual_Change"] < 0]["Actual_Change"]


    # Quantiles based only on actuals
    pos_low, pos_high = pos.quantile([0.33, 0.66])
    neg_low, neg_high = neg.quantile([0.33, 0.66])

    results_df["Actual_LMH_Dir"] = results_df["Actual_Change"].apply(
        lambda x: directional_lmh(x, pos_low, pos_high, neg_low, neg_high)
    )
    results_df["Predicted_LMH_Dir"] = results_df["Predicted_Change"].apply(
        lambda x: directional_lmh(x, pos_low, pos_high, neg_low, neg_high)
    )

    # ---- 6-class confusion matrix ----
    labels = ["Bull_Low", "Bull_Medium", "Bull_High", "Bear_Low", "Bear_Medium", "Bear_High"]
    cm = confusion_matrix(results_df["Actual_LMH_Dir"], results_df["Predicted_LMH_Dir"], labels=labels)
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)

    # print("6-Class Confusion Matrix:")
    # print(cm_df)

    # ---- Directional accuracy per class ----
    directional_per_class = {}
    for label in labels:
        subset = results_df[results_df["Actual_LMH_Dir"] == label]
        if len(subset) > 0:
            directional_per_class[label] = (subset["Actual_LMH_Dir"] == subset["Predicted_LMH_Dir"]).mean()
        else:
            directional_per_class[label] = np.nan  # handle empty classes

    # print("\nDirectional Accuracy per Class:")
    # for k, v in directional_per_class.items():
    #     print(f"{k}: {v:.2%}" if not np.isnan(v) else f"{k}: N/A")
    
    actual_counts = results_df["Actual_LMH_Dir"].value_counts().reindex(labels, fill_value=0)
    pred_counts = results_df["Predicted_LMH_Dir"].value_counts().reindex(labels, fill_value=0)

    counts_df = pd.DataFrame({
        "Actual_Count": actual_counts,
        "Predicted_Count": pred_counts
    })

    print("Counts per Class:")
    print(counts_df)


    # ---- Out-of-sample R² ----
    mean_train = model.model.endog.mean() # it should technically be the variable below since this is OOS, redoo graphs later # 
    # mean_train = y.iloc[:int(len(y) * 0.80)].mean() 
    sse_model = results_df["Squared_Error"].sum() 
    sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum() 
    r2_oos = 1 - (sse_model / sse_mean)

    results_df.drop(['Correct_Direction', 'Predicted_Direction', 'Squared_Error', 'Actual_Change', 'Actual_Direction'], axis=1, inplace=True)
    # print(results_df)

    print(f"Out-of-sample R²: {r2_oos:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")
    # print(results_df["Actual"].max())

    return results_df, directional_accuracy, r2_oos

def directional_lmh(change, pos_low, pos_high, neg_low, neg_high):
    '''
    Classifies a monthly change into one of six directional-magnitude categories based on actual return thresholds.
    Split into Low, medium, and high, it is based according to the actual market movement.
    '''
    if change > 0:  # Bull
        if change <= pos_low:
            return "Bull_Low"
        elif change <= pos_high:
            return "Bull_Medium"
        else:
            return "Bull_High"
    elif change < 0:  # Bear
        if change >= neg_high:  # closer to 0 → low magnitude
            return "Bear_Low"
        elif change >= neg_low:
            return "Bear_Medium"
        else:
            return "Bear_High"
    else:
        return "Neutral"
    
def graph(df, train, test, y_pred, etf, output_dir, directional_accuracy, r2_oos):
    """
    Plots:
    1. Train + Test + Predicted (full time series view)
    2. Test Actual vs Predicted (evaluation view)
    Saves figure to disk.
    """
    plt.figure(figsize=(14, 6))

    # -------------------------
    # Left Plot: Full Context
    # -------------------------
    ax1 = plt.subplot(1, 2, 1)
    ax1.plot(train.index, train["y"], label="Train (Actual)", color="black")
    ax1.plot(test.index, test["y"], label="Test (Actual)", linewidth=2, color="orange")
    ax1.plot(test.index, y_pred, label="Test (Predicted)", linestyle="--", color="green")
    
    ax1.axvline(test.index[0], color="black", linestyle=":", label="Train/Test Split")
    ax1.set_title("Train/Test with Predictions")
    ax1.set_xlabel("Month")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True)

    # -------------------------
    # Right Plot: Evaluation
    # -------------------------
    ax2 = plt.subplot(1, 2, 2)
    ax2.plot(df.index, df["Actual"], label="Actual", linewidth=2, color="orange")
    ax2.plot(df.index, df["Predicted"], label="Predicted", linestyle="--", color="green")
    ax2.set_title("Out-of-Sample: Actual vs Predicted")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True)

    # ---- Add Metrics Box ----
    metrics_text = (
        f"R² (OOS): {r2_oos:.4f}\n"
        f"Directional Accuracy: {directional_accuracy:.2%}"
    )

    ax2.text(
        0.02, 0.98,               # top-left corner
        metrics_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8)
    )

    plt.tight_layout()

    # Save
    os.makedirs(output_dir, exist_ok=True)
    etf_name = os.path.basename(etf).replace(".csv", "")
    filepath = os.path.join(output_dir, f"{etf_name}_results.png")
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    # Comment this out to not display graph
    plt.show()
    plt.close()