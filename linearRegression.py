import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm
import os


    
def linear_regression(x, y, etf, output_dir="reports/images"):
    os.makedirs(output_dir, exist_ok=True)

    df = x.copy()
    df["y"] = y

    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    predictors = " + ".join(x.columns)
    formula = f"y ~ {predictors}"

    model = smf.ols(formula, data=train).fit()

    y_pred = model.predict(test)

    results, directional_accuracy, r2_oos = model_testing(x, y, model, test)
    
    graph(results, train, test, y_pred, etf, output_dir, directional_accuracy, r2_oos)

    return model.summary(), anova_lm(model, typ=1)


def model_testing(x, y, model, test):
    """
    For a fitted regression model and a test set, calculates
    predicted vs actual for each month individually, 
    """
    test = test.copy()
    results = []

    for idx, row in test.iterrows():
        x_row = row[x.columns]  # predictor values for this month
        y_actual = row["y"]
        
        # Predict for this single month
        y_pred = model.predict(row.to_frame().T).iloc[0]

        results.append({
            "Month": idx,
            "Actual": y_actual,
            "Predicted": y_pred
        })

    results_df = pd.DataFrame(results)
    results_df.set_index("Month", inplace=True)

    # Optional: add error columns
    results_df["Error"] = results_df["Actual"] - results_df["Predicted"]
    results_df["Squared_Error"] = results_df["Error"] ** 2
    
    # Calculate month-over-month change
    results_df = results_df.copy()
    results_df["Actual_Direction"] = results_df["Actual"].diff() > 0
    results_df["Predicted_Direction"] = results_df["Predicted"].diff() > 0

    # Correct if direction matches
    results_df["Correct_Direction"] = results_df["Actual_Direction"] == results_df["Predicted_Direction"]

    # Optional: convert to Yes / No
    results_df["Direction_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

    # Drop the first month since diff() creates NaN
    results_df = results_df.iloc[1:]

    # Convert boolean to Yes / No
    results_df["Direction_Correct_Label"] = results_df["Correct_Direction"].map({True: "Yes", False: "No"})

    # Example: show month-by-month classification
    print(results_df[["Actual", "Predicted", "Direction_Correct_Label"]])

    # out of sample r2
    directional_accuracy = results_df["Correct_Direction"].mean()
    mean_train = model.model.endog.mean()
    sse_model = results_df["Squared_Error"].sum()
    sse_mean = ((results_df["Actual"] - mean_train) ** 2).sum()
    r2_oos = 1 - (sse_model / sse_mean)

    print(f"Out-of-sample R²: {r2_oos:.4f}")
    print(f"Directional Accuracy: {directional_accuracy:.2%}")

    return results_df, directional_accuracy, r2_oos

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
    plt.close()