import streamlit as st
import glob
import os

from main import create_linear_model
from data_cleanse import *

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

selected_etf = st.selectbox(
    "Select ETF",
    etf_files,
    format_func=lambda x: os.path.basename(x).replace(".csv", "")
)

# ----------------------------
# Macro Selection
# ----------------------------

macro_folder = "data/raw_data"
macro_files = [
    f for f in glob.glob(os.path.join(macro_folder, "*.csv"))
    if "ETFs" not in f
]

selected_macros = st.multiselect(
    "Select Macro CSVs",
    macro_files,
    format_func=lambda x: os.path.basename(x).replace(".csv", "")
)

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

corr_threshold = st.slider(
    "PCA Correlation Threshold",
    min_value=0.5,
    max_value=0.95,
    value=0.80,
    step=0.05
)

variance_explained = st.slider(
    "Variance Explained",
    min_value=0.70,
    max_value=0.99,
    value=0.90,
    step=0.01
)

stability_threshold = st.slider(
    "Lag Stability Threshold",
    min_value=0.30,
    max_value=0.90,
    value=0.50,
    step=0.05
)

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
            corr_threshold=corr_threshold,
            variance_explained=variance_explained,
            stability_threshold=stability_threshold,
            display=False
        )
        print(TABLE_CONFIG)

    # ----------------------------
    # Display Results
    # ----------------------------

    st.success("Model Completed")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("OLS Summary")
        st.text(osl)

    with col2:
        st.subheader("ANOVA Table")
        st.dataframe(anova)

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