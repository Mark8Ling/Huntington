import glob
import os

import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
import altair as alt
import pandas as pd
import streamlit as st
from statsmodels.tsa.arima.model import ARIMA

from data_cleanse import (
    MoM,
    YoY,
    diff,
    enforce_stationary,
    interpolate_monthly,
    log_diff,
    read_csv_standard,
    read_quarterly,
    fix_pd,
    master_table,
)
from main import create_linear_model
from random_forest import run_random_forest
from sector_volatility import run_risk_engine


st.set_page_config(page_title="Huntington ETF Macro Lab", layout="wide")

PROCESSING = {
    "read": read_csv_standard,
    "quarterly": read_quarterly,
    "MoM": MoM,
    "interpolate_monthly": interpolate_monthly,
    "YoY": YoY,
    "enforce_stationary": enforce_stationary,
    "log_diff": log_diff,
    "diff": diff,
}

PIPELINE_STEPS = [
    ("Quarterly Resample", "quarterly"),
    ("Interpolate Monthly", "interpolate_monthly"),
    ("Log Diff", "log_diff"),
    ("Diff", "diff"),
    ("YoY % Change", "YoY"),
    ("MoM % Change", "MoM"),
    ("Enforce Stationary", "enforce_stationary"),
]

ETF_INDUSTRY_MAP = {
    "XLB": "Materials",
    "XLC": "Communication Services",
    "XLE": "Energy",
    "XLF": "Financials",
    "XLI": "Industrials",
    "XLK": "Technology",
    "XLP": "Consumer Staples",
    "XLRE": "Real Estate",
    "XLU": "Utilities",
    "XLV": "Healthcare",
    "XLY": "Consumer Discretionary",
}

MACRO_DISPLAY_MAP = {
    "APU000072610": "Electricity Price",
    "CES1021100001": "Oil & Gas Sector Employment",
    "CPIENGSL": "Energy Price Index",
    "DTWEXBGS": "US Dollar Trade Index",
    "FEDFUNDS": "Federal Funds Rate",
    "GASREGCOVM": "Gasoline Price",
    "GDP": "GDP (Nominal)",
    "GDPC1": "GDP (Real, Chained 2017 $)",
    "GS10": "10-Year Treasury Yield",
    "INDPRO": "Industrial Production Index",
    "IPG211111CN": "Oil & Gas Production Index",
    "MCOILWTICO": "WTI Oil Price",
    "MHHNGSP": "Natural Gas Price",
    "PCEPI": "PCE Price Index",
    "PPIENG": "Producer Price Index (Energy)",
    "TOTALSA": "Total Vehicle Sales",
    "TTLCONS": "Total Construction Spending",
    "UMCSENT": "Consumer Sentiment Index",
    "UNRATE": "Unemployment Rate",
}


def etf_display_name(path):
    base = os.path.basename(path)
    code = base.replace("_monthly.csv", "").replace(".csv", "")
    industry = ETF_INDUSTRY_MAP.get(code, code)
    return f"{industry} ({code})"


def macro_display_name(path_or_code):
    base = os.path.basename(path_or_code)
    code = base.replace(".csv", "")
    return MACRO_DISPLAY_MAP.get(code, code)


def list_etf_files(folder):
    return sorted(glob.glob(os.path.join(folder, "*.csv")))


def list_macro_files(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    return sorted([f for f in files if "ETFs" not in f])


def etf_ticker_from_path(path):
    base = os.path.basename(path)
    return base.replace("_monthly.csv", "").replace(".csv", "")


@st.cache_data(show_spinner=False)
def run_model_cached(
    table_config,
    etf_path,
    use_lag,
    use_pca,
    corr_threshold,
    variance_explained,
    stability_threshold,
    generate_plot,
):
    return create_linear_model(
        PROCESSING=PROCESSING,
        TABLE_CONFIG=table_config,
        etf=etf_path,
        use_lag=use_lag,
        use_pca=use_pca,
        corr_threshold=corr_threshold,
        variance_explained=variance_explained,
        stability_threshold=stability_threshold,
        display=False,
        generate_plot=generate_plot,
        show_plot=False,
        return_results=True,
    )


@st.cache_data(show_spinner=False)
def run_arima_cached(table_config, etf_path, arima_order, train_ratio):
    X = master_table(table_config, PROCESSING, "arima_macros")
    y = fix_pd(etf_path)["Close"].pct_change().dropna()

    y, X = y.align(X, join="inner")
    if len(y) < 12:
        raise ValueError("Not enough data after alignment for ARIMA training.")

    train_size = int(len(y) * train_ratio)
    y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
    X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]

    model = ARIMA(endog=y_train, exog=X_train, order=arima_order)
    model_fit = model.fit()

    forecast = model_fit.get_forecast(steps=len(y_test), exog=X_test)
    y_pred = forecast.predicted_mean
    forecast_ci = forecast.conf_int()

    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    oos_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    y_test_diff = y_test.diff().dropna()
    y_pred_diff = y_pred.diff().dropna()
    y_test_diff, y_pred_diff = y_test_diff.align(y_pred_diff, join="inner")
    if len(y_test_diff) == 0:
        directional_accuracy = float("nan")
    else:
        directional_accuracy = (y_test_diff.sign() == y_pred_diff.sign()).mean()

    results_df = pd.DataFrame(
        {
            "Actual": y_test,
            "Predicted": y_pred,
        }
    )

    return model_fit.summary().as_text(), results_df, forecast_ci, oos_r2, directional_accuracy


@st.cache_data(show_spinner=False)
def run_random_forest_cached(
    table_config,
    etf_path,
    n_estimators,
    max_depth,
    min_sample_splits,
    max_features,
):
    X = master_table(table_config, PROCESSING, "rf_macros")
    y = fix_pd(etf_path)["Close"].pct_change().dropna()

    y, X = y.align(X, join="inner")
    if len(y) < 30:
        raise ValueError("Not enough data after alignment for Random Forest.")

    df = pd.concat([y.rename("Close"), X], axis=1).dropna()
    regressor, X_test, y_test, metrics = run_random_forest(
        df,
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_sample_splits=min_sample_splits,
        max_features=max_features,
    )

    y_pred = pd.Series(regressor.predict(X_test), index=y_test.index)
    ss_res = ((y_test - y_pred) ** 2).sum()
    ss_tot = ((y_test - y_test.mean()) ** 2).sum()
    oos_r2 = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    results_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    return regressor, X_test, y_test, metrics, results_df, oos_r2


st.title("ETF Macro Regression Builder")

with st.sidebar:
    st.header("Data")

    etf_files = list_etf_files("data/raw_data/ETFs")
    if not etf_files:
        st.error("No ETF CSV files found.")
        st.stop()

    selected_etf = st.selectbox(
        "ETF",
        etf_files,
        format_func=etf_display_name,
    )

    macro_files = list_macro_files("data/raw_data")
    selected_macros = st.multiselect(
        "Macro CSVs",
        macro_files,
        format_func=macro_display_name,
    )

    st.divider()
    st.header("Processing")

    use_same_pipeline = st.checkbox("Use same pipeline for all macros", value=False)
    pipeline_labels = [label for label, _ in PIPELINE_STEPS]
    pipeline_values = [value for _, value in PIPELINE_STEPS]

    global_pipeline = []
    if use_same_pipeline:
        global_pipeline = st.multiselect(
            "Pipeline (order fixed)",
            pipeline_values,
            default=[],
            format_func=lambda x: dict(PIPELINE_STEPS).get(x, x),
        )

    st.divider()
    st.header("Model Options")

    use_lag = st.checkbox("Apply Optimal Lag Engine", value=True)
    use_pca = st.checkbox("Apply PCA", value=True)

    corr_threshold = st.slider(
        "PCA Correlation Threshold",
        min_value=0.5,
        max_value=0.95,
        value=0.80,
        step=0.05,
    )

    variance_explained = st.slider(
        "Variance Explained",
        min_value=0.70,
        max_value=0.99,
        value=0.90,
        step=0.01,
    )

    stability_threshold = st.slider(
        "Lag Stability Threshold",
        min_value=0.30,
        max_value=0.90,
        value=0.50,
        step=0.05,
    )

    generate_plot = st.checkbox("Generate PNG Plot", value=False)

    if st.button("Refresh cache"):
        st.cache_data.clear()
        st.success("Cache cleared.")


if not selected_macros:
    st.info("Select at least one macro to enable Regression/ARIMA/Random Forest.")


table_config = {}
if selected_macros:
    with st.expander("Macro Pipelines", expanded=True):
        for macro_path in selected_macros:
            macro_name = os.path.basename(macro_path).replace(".csv", "")
            macro_label = macro_display_name(macro_name)

            if use_same_pipeline:
                pipeline = ["read"] + global_pipeline
            else:
                pipeline = ["read"]
                selected_steps = st.multiselect(
                    f"{macro_label} pipeline",
                    pipeline_values,
                    default=[],
                    key=f"pipeline_{macro_name}",
                    format_func=lambda x: dict(PIPELINE_STEPS).get(x, x),
                )
                pipeline.extend(selected_steps)

            table_config[macro_name] = {
                "path": macro_path,
                "pipeline": pipeline,
                "shift": 0,
            }


regression_tab, arima_tab, risk_tab, rf_tab = st.tabs(
    ["Regression", "ARIMA", "Sector Risk", "Random Forest"]
)

with regression_tab:
    if not selected_macros:
        st.info("Select at least one macro to run regression.")
    else:
        run = st.button("Run Model", key="run_regression")

        if run:
            with st.spinner("Running macro regression pipeline..."):
                (
                    ols_summary,
                    anova_df,
                    valid_lag,
                    results_df,
                    directional_accuracy,
                    r2_oos,
                ) = run_model_cached(
                    table_config,
                    selected_etf,
                    use_lag,
                    use_pca,
                    corr_threshold,
                    variance_explained,
                    stability_threshold,
                    generate_plot,
                )

            st.success("Model completed")

            metrics_col1, metrics_col2 = st.columns(2)
            metrics_col1.metric("Out-of-sample R2", f"{r2_oos:.4f}")
            metrics_col2.metric("Directional Accuracy", f"{directional_accuracy:.2%}")

            plot_df = results_df.reset_index()
            date_col = "Month" if "Month" in plot_df.columns else plot_df.columns[0]
            plot_df = plot_df.rename(columns={date_col: "Date"})

            long_df = plot_df.melt(
                id_vars=["Date"],
                value_vars=["Actual", "Predicted"],
                var_name="Series",
                value_name="Value",
            )

            chart = (
                alt.Chart(long_df)
                .mark_line()
                .encode(
                    x=alt.X("Date:T", title="Date"),
                    y=alt.Y("Value:Q", title="Value"),
                    color=alt.Color("Series:N", title=""),
                    tooltip=["Date:T", "Series:N", "Value:Q"],
                )
                .properties(height=360)
            )

            st.subheader("Actual vs Predicted")
            st.altair_chart(chart, use_container_width=True)

            st.subheader("Results Table")
            st.dataframe(results_df, use_container_width=True)

            csv_bytes = results_df.to_csv().encode("utf-8")
            etf_name = os.path.basename(selected_etf).replace(".csv", "")
            st.download_button(
                "Download results CSV",
                data=csv_bytes,
                file_name=f"{etf_name}_results.csv",
                mime="text/csv",
            )

            if valid_lag:
                st.subheader("Valid Lags Applied")
                lag_df = pd.DataFrame(valid_lag, columns=["Macro", "Lag", "Stability"])
                lag_df["Macro"] = lag_df["Macro"].map(macro_display_name)
                st.dataframe(lag_df, use_container_width=True)

            with st.expander("OLS Summary", expanded=False):
                st.text(ols_summary)

            with st.expander("ANOVA Table", expanded=False):
                st.dataframe(anova_df, use_container_width=True)

            if generate_plot:
                image_path = os.path.join("reports", "images", f"{etf_name}_results.png")
                if os.path.exists(image_path):
                    st.subheader("Train/Test Regression Plot")
                    st.image(image_path, use_container_width=True)
                else:
                    st.warning(f"Plot image not found at {image_path}")

with arima_tab:
    if not selected_macros:
        st.info("Select at least one macro to run ARIMA/ARIMAX.")
    else:
        st.subheader("ARIMA / ARIMAX")

        arima_col1, arima_col2, arima_col3, arima_col4 = st.columns(4)
        with arima_col1:
            p = st.number_input("AR Order (p)", min_value=0, max_value=6, value=1, step=1)
        with arima_col2:
            d = st.number_input("Diff (d)", min_value=0, max_value=3, value=1, step=1)
        with arima_col3:
            q = st.number_input("MA Order (q)", min_value=0, max_value=6, value=1, step=1)
        with arima_col4:
            train_ratio = st.slider(
                "Train Split",
                min_value=0.60,
                max_value=0.95,
                value=0.80,
                step=0.05,
            )

        run_arima = st.button("Run ARIMA", key="run_arima")

        if run_arima:
            try:
                with st.spinner("Running ARIMA model..."):
                    (
                        arima_summary,
                        arima_results,
                        arima_ci,
                        arima_r2,
                        arima_directional_accuracy,
                    ) = run_arima_cached(
                        table_config,
                        selected_etf,
                        (int(p), int(d), int(q)),
                        float(train_ratio),
                    )

                st.success("ARIMA completed")

                arima_metrics_col1, arima_metrics_col2 = st.columns(2)
                arima_metrics_col1.metric("Out-of-sample R2", f"{arima_r2:.4f}")
                arima_metrics_col2.metric(
                    "Directional Accuracy",
                    f"{arima_directional_accuracy:.2%}"
                    if isinstance(arima_directional_accuracy, (int, float))
                    else "n/a",
                )

                arima_plot_df = arima_results.reset_index()
                date_col = (
                    "Month" if "Month" in arima_plot_df.columns else arima_plot_df.columns[0]
                )
                arima_plot_df = arima_plot_df.rename(columns={date_col: "Date"})

                arima_long_df = arima_plot_df.melt(
                    id_vars=["Date"],
                    value_vars=["Actual", "Predicted"],
                    var_name="Series",
                    value_name="Value",
                )

                arima_chart = (
                    alt.Chart(arima_long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Series:N", title=""),
                        tooltip=["Date:T", "Series:N", "Value:Q"],
                    )
                    .properties(height=360)
                )

                if arima_ci is not None:
                    date_col_name = arima_ci.index.name or "index"
                    arima_ci_plot = arima_ci.reset_index().rename(
                        columns={
                            date_col_name: "Date",
                            arima_ci.columns[0]: "Lower",
                            arima_ci.columns[1]: "Upper",
                        }
                    )
                    band = (
                        alt.Chart(arima_ci_plot)
                        .mark_area(opacity=0.2)
                        .encode(
                            x=alt.X("Date:T"),
                            y=alt.Y("Lower:Q"),
                            y2=alt.Y2("Upper:Q"),
                        )
                    )
                    arima_chart = band + arima_chart

                st.subheader("Actual vs Predicted")
                st.altair_chart(arima_chart, use_container_width=True)

                st.subheader("Results Table")
                st.dataframe(arima_results, use_container_width=True)

                arima_csv = arima_results.to_csv().encode("utf-8")
                etf_name = os.path.basename(selected_etf).replace(".csv", "")
                st.download_button(
                    "Download ARIMA results CSV",
                    data=arima_csv,
                    file_name=f"{etf_name}_arima_results.csv",
                    mime="text/csv",
                )

                with st.expander("ARIMA Summary", expanded=False):
                    st.text(arima_summary)
            except Exception as exc:
                st.error(f"ARIMA failed: {exc}")

with risk_tab:
    st.subheader("Sector Risk Engine")
    risk_ticker = etf_ticker_from_path(selected_etf)
    st.caption(f"Selected ETF: {risk_ticker}")

    run_risk = st.button("Run Sector Risk Engine", key="run_sector_risk")
    if run_risk:
        try:
            with st.spinner("Computing sector risk..."):
                risk_data = run_risk_engine(risk_ticker)

            if "error" in risk_data:
                st.error(f"Sector risk failed: {risk_data['error']}")
            else:
                risk_score = risk_data.get("risk_score", float("nan"))
                if risk_score < 0.30:
                    risk_label = "Low Risk"
                elif risk_score < 0.60:
                    risk_label = "Moderate Risk"
                elif risk_score < 0.80:
                    risk_label = "High Risk"
                else:
                    risk_label = "Very High Risk"

                st.metric("Risk Score", f"{risk_score:.2f}", risk_label)

                st.subheader("Risk Components")
                st.json(risk_data)
        except Exception as exc:
            st.error(f"Sector risk failed: {exc}")

with rf_tab:
    if not selected_macros:
        st.info("Select at least one macro to run Random Forest.")
    else:
        st.subheader("Random Forest Regression")
        rf_col1, rf_col2, rf_col3, rf_col4 = st.columns(4)
        with rf_col1:
            n_estimators = st.number_input(
                "Trees",
                min_value=50,
                max_value=1000,
                value=200,
                step=50,
            )
        with rf_col2:
            max_depth = st.number_input(
                "Max Depth (0 = None)",
                min_value=0,
                max_value=50,
                value=0,
                step=1,
            )
        with rf_col3:
            min_samples_split = st.number_input(
                "Min Samples Split",
                min_value=2,
                max_value=20,
                value=2,
                step=1,
            )
        with rf_col4:
            max_features = st.selectbox(
                "Max Features",
                options=["sqrt", "log2", None],
                format_func=lambda x: "None" if x is None else x,
            )

        run_rf = st.button("Run Random Forest", key="run_rf")
        if run_rf:
            try:
                with st.spinner("Training Random Forest..."):
                    (
                        rf_model,
                        rf_X_test,
                        rf_y_test,
                        rf_metrics,
                        rf_results,
                        rf_r2,
                    ) = run_random_forest_cached(
                        table_config,
                        selected_etf,
                        int(n_estimators),
                        None if int(max_depth) == 0 else int(max_depth),
                        int(min_samples_split),
                        max_features,
                    )

                st.success("Random Forest completed")
                st.metric("Out-of-sample R2", f"{rf_r2:.4f}")

                rf_plot_df = rf_results.reset_index()
                date_col = "Month" if "Month" in rf_plot_df.columns else rf_plot_df.columns[0]
                rf_plot_df = rf_plot_df.rename(columns={date_col: "Date"})

                rf_long_df = rf_plot_df.melt(
                    id_vars=["Date"],
                    value_vars=["Actual", "Predicted"],
                    var_name="Series",
                    value_name="Value",
                )

                rf_chart = (
                    alt.Chart(rf_long_df)
                    .mark_line()
                    .encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Value:Q", title="Value"),
                        color=alt.Color("Series:N", title=""),
                        tooltip=["Date:T", "Series:N", "Value:Q"],
                    )
                    .properties(height=360)
                )

                st.subheader("Actual vs Predicted")
                st.altair_chart(rf_chart, use_container_width=True)

                st.subheader("Results Table")
                st.dataframe(rf_results, use_container_width=True)

                feat_imp = pd.DataFrame(
                    {
                        "Feature": rf_X_test.columns,
                        "Importance": rf_model.feature_importances_,
                    }
                ).sort_values("Importance", ascending=False)

                feat_chart = (
                    alt.Chart(feat_imp)
                    .mark_bar()
                    .encode(
                        x=alt.X("Importance:Q"),
                        y=alt.Y("Feature:N", sort="-x"),
                        tooltip=["Feature:N", "Importance:Q"],
                    )
                    .properties(height=360)
                )

                st.subheader("Feature Importance")
                st.altair_chart(feat_chart, use_container_width=True)

                rf_metrics_df = pd.DataFrame(rf_metrics)
                with st.expander("Time Split Metrics", expanded=False):
                    st.dataframe(rf_metrics_df, use_container_width=True)
            except Exception as exc:
                st.error(f"Random Forest failed: {exc}")
