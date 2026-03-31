import streamlit as st
import glob
import os
from main import create_linear_model
from data_cleanse import *
# Cut out the stats jargon


st.set_page_config(layout="wide")

# ----------------------------
# Title
# ----------------------------
st.title("ETF Macro Regression Builder")

# ----------------------------
# ETF Selection
# ----------------------------

etf_folder = "data/raw_data/ETFs"
etf_files = glob.glob(os.path.join(etf_folder, "*.csv"))

if not etf_files:
    st.error("No ETF CSV files found.")
    st.stop()

etf_map = {
    "Energy": "data/raw_data/ETFs/XLE_monthly.csv",
    "Utilities": "data/raw_data/ETFs/XLU_monthly.csv",
    "Financials": "data/raw_data/ETFs/XLF_monthly.csv",
    "Healthcare": "data/raw_data/ETFs/XLV_monthly.csv"
}

selected_sector = st.selectbox(
    "Select Sector",
    list(etf_map.keys())
)

selected_etf = etf_map[selected_sector]
# ----------------------------
# Macro Selection
# ----------------------------
macro_map = {
    "Oil": "data/raw_data/MCOILWTICO.csv",
    "GDP": "data/raw_data/GDP.csv",
    "Unemployment Rate": "data/raw_data/UNRATE.csv",
    "Interest Rates": "data/raw_data/FEDFUNDS.csv",
    "Inflation": "data/raw_data/PCEPI.csv"
}

selected_macro_names = st.multiselect(
    "Select Macros",
    list(macro_map.keys())
)

selected_macros = [macro_map[name] for name in selected_macro_names]

# ----------------------------
# Processing Options
# ----------------------------

st.subheader("Macro Processing Options")

apply_log_diff = st.checkbox("Apply Log Diff")
apply_yoy = st.checkbox("Apply YoY")
apply_mom = st.checkbox("Apply MoM")
apply_stationary = st.checkbox("Enforce Stationarity")
apply_interpolate = st.checkbox("Interpolate Monthly")

# ----------------------------
# Model Options
# ----------------------------

st.subheader("Model Options")

use_lag = st.checkbox("Apply Optimal Lag Engine", value=True)
use_pca = st.checkbox("Apply PCA", value=True)


# ----------------------------
# Run Model
# ----------------------------

if st.button("Run Model"):

    if not selected_macros:
        st.warning("Please select at least one macro variable.")
        st.stop()

    with st.spinner("Running macro regression pipeline..."):

        PROCESSING = {
            "read": read_csv_standard,
            "quarterly": read_quarterly,
            "MoM": MoM,
            "interpolate_monthly": interpolate_monthly,
            "YoY": YoY,
            "enforce_stationary": enforce_stationary,
            "log_diff": log_diff
        }

        TABLE_CONFIG = {}

        for macro_path in selected_macros:

            name = os.path.basename(macro_path).replace(".csv", "")
            pipeline = ["read"]

            if apply_interpolate:
                pipeline.append("interpolate_monthly")

            if apply_log_diff:
                pipeline.append("log_diff")

            if apply_yoy:
                pipeline.append("YoY")

            if apply_mom:
                pipeline.append("MoM")

            if apply_stationary:
                pipeline.append("enforce_stationary")

            TABLE_CONFIG[name] = {
                "path": macro_path,
                "pipeline": pipeline,
                "shift": 0
            }
        

        osl, anova, valid_lag = create_linear_model(
            PROCESSING=PROCESSING,
            TABLE_CONFIG=TABLE_CONFIG,
            etf=selected_etf,
            use_lag=use_lag,
            use_pca=use_pca,
            corr_threshold=0.8,
            variance_explained=0.9,
            stability_threshold=0.5,
            display=False
        )
    # ----------------------------
    # Display Results
    # ----------------------------

    st.success("Model Completed")

    col1, col2 = st.columns(2)

    ols_table = osl.tables[1]
    ols_df = pd.DataFrame(ols_table.data[1:], columns=ols_table.data[0])

    # First column contains variable names
    ols_df = ols_df.rename(columns={ols_df.columns[0]: "variable"})
    ols_df = ols_df.set_index("variable")

    # normalize column names
    ols_df.columns = [col.lower() for col in ols_df.columns]

    # Keep only what you want
    cols_to_keep = ['coef', 't', 'p>|t|']
    ols_df = ols_df[[c for c in cols_to_keep if c in ols_df.columns]]

    # Replace macro file names with friendly names
    name_map = {os.path.basename(path).replace(".csv",""): name 
                for name, path in macro_map.items()}

    ols_df.index = [name_map.get(idx, idx) for idx in ols_df.index]

    with col1:
        st.subheader("Regression Summary (Simplified)")
        st.dataframe(ols_df)

        st.subheader("Valid Lags Applied")

    if valid_lag:
        for col, lag, stability in valid_lag:
            st.write(f"{col} → Lag {lag} (Stability: {stability:.2f})")
    else:
        st.write("No lags applied.")

    # ----------------------------
    # Display Regression Plot
    # ----------------------------

    etf_name = os.path.basename(selected_etf).replace(".csv", "")
    image_path = f"reports/images/{etf_name}_results.png"

    if os.path.exists(image_path):
        st.subheader("Train/Test Regression Plot")
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"Plot image not found at {image_path}")