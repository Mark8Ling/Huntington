""" Provides visuals for the risk_engine output, showing rankings among sectors from a number of sectors """
import pandas as pd
import textwrap
import matplotlib.pyplot as plt

def visualize_by_risk(sector_risk_data: dict[dict], etf_ticker):
    """ Ranks each sector based upon their relative risk score """
    
    risk = "risk_score"
    
    # sort by risk score
    risk_table = rank_by_risk(sector_risk_data, risk)

    fig, ax = plt.subplots(figsize=(20, 4))
    ax.set_title(
        "Sectors Ranked by Relative Risk Score",
        fontsize=12,
        fontweight='bold',
        pad=5
    )
    ax.axis('off')

    wrapped_columns = [
        "\n".join(textwrap.wrap(col, width=12))
        for col in risk_table.columns
    ]

    table = ax.table(
        cellText=risk_table.values,
        colLabels=wrapped_columns,
        loc='center',
        cellLoc='center',
        bbox=[0, 0, 1, 0.85]
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.9)

    # column index for risk score highlight
    risk_col_idx = list(risk_table.columns).index("Risk Score")

    # find etf row index
    highlight_row = risk_table.index[
        risk_table["Sector"] == etf_ticker
    ]
    highlight_row = highlight_row[0] if len(highlight_row) > 0 else None

    for (row, col), cell in table.get_celld().items():

        # header
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2E86C1')

        else:
            data_row_idx = row - 1

            # highlight etf row first
            if data_row_idx == highlight_row:
                cell.set_facecolor("#AEF1C0")  # light blue row highlight
                cell.set_text_props(weight='bold')

            # Then apply column logic
            elif col == risk_col_idx:
                cell.set_facecolor("#D6F8DC")  # risk score column highlight

            else:
                cell.set_facecolor('#F2F3F4' if row % 2 == 0 else 'white')

    plt.show()

def rank_by_risk(sector_risk_data: dict[dict], risk: str):
    # sort by risk score
    sorted_tickers = sorted(
        sector_risk_data.items(),
        key=lambda ticker: ticker[1][risk],
        reverse=True
    )

    # create df for generating output table
    df = pd.DataFrame([
    {
        "Sector": ticker,
        "Risk Score": data["risk_score"],

        "Volatility": data["volatility"],
        "Normalized Volatility": data["normalized_volatility"],

        "Beta": data["beta"],
        "Normalized Beta": data["normalized_beta"],

        "Holdings Correlation": data["holdings_correlation"],
        "Normalized Correlations": data["normalized_correlations"],
    }
        for ticker, data in sorted_tickers
    ])
    numeric_cols = df.select_dtypes(include="number").columns
    df[numeric_cols] = df[numeric_cols].round(2)

    return df

def visualize_by_metric(sector_risk_data: dict[dict], etf_ticker, metric: str):
    # sort by metric
    sorted_metric_df = rank_by_metric(sector_risk_data, metric)

    # visualize by metric
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.set_title(f"Sectors Sorted by {metric.capitalize()}", fontsize=14, fontweight='bold', pad=15)
    ax.axis("off")

    table = ax.table(
        cellText=sorted_metric_df.values,
        colLabels=sorted_metric_df.columns,
        loc='center',
        cellLoc='center'
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Find etf ticker row to highlight 
    highlight_row = sorted_metric_df.index[
        sorted_metric_df["Sector"] == etf_ticker
    ]
    highlight_row = highlight_row[0] if len(highlight_row) > 0 else None

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2E86C1')

        else:
            # adjust because table row 0 = header, so data starts at row 1
            data_row_idx = row - 1

            if data_row_idx == highlight_row:
                cell.set_facecolor("#AEF1B4")  # highlight etf row
                cell.set_text_props(weight='bold')
            else:
                cell.set_facecolor('#F2F3F4' if row % 2 == 0 else 'white')

    plt.show()

def rank_by_metric(sector_risk_data: dict[dict], metric: str):
    """ Given a metric sort values from highest to lowest """
    
    # sort all the etf's by the given metric (i.e, beta, volatility, normalized_beta, etc.)
    sorted_metrics = sorted(
        sector_risk_data.items(),
        key=lambda ticker: ticker[1][metric],
        reverse=True
    )
    
    if metric != "holdings_correlation":
        # pull specific metric data
        metric_df = pd.DataFrame([
            {
                "Sector": ticker,
                metric: data[metric],
                f"normalized_{metric}": data[f"normalized_{metric}"]
            }
            for ticker, data in sorted_metrics
        ])
        
        # rounding
        metric_df[metric] = metric_df[metric].round(3)
        metric_df[f"normalized_{metric}"] = metric_df[f"normalized_{metric}"].round(3)
    
    else:
      # pull specific metric data
        metric_df = pd.DataFrame([
            {
                "Sector": ticker,
                metric: data[metric],
                f"normalized_correlations": data[f"normalized_correlations"]
            }
            for ticker, data in sorted_metrics
        ])  
    
        # rounding
        metric_df[metric] = metric_df[metric].round(3)
        metric_df[f"normalized_correlations"] = metric_df[f"normalized_correlations"].round(3)

    # capitalize
    metric_df.columns = metric_df.columns.str.capitalize()
    
    return metric_df

def visualize_holdings(etf_ticker):
    # get top holdings
    top_holdings = get_holdings(etf_ticker)
    top_holdings['Weight'] = top_holdings['Weight'].round(2)

    # visualize top holdings
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f"{etf_ticker} Top Holdings", fontsize=14, fontweight='bold')
    ax.axis('off')

    table = ax.table(
        cellText=top_holdings.values,
        colLabels=top_holdings.columns,
        loc='center',
        cellLoc='center'
    )

    # Style
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.6)

    # Header styling
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#2E86C1')  # blue
        else:
            # alternating row colors
            cell.set_facecolor('#F2F3F4' if row % 2 == 0 else 'white')

    plt.show()

def get_holdings(etf_ticker):
    sector_info = pd.read_excel(rf'data\raw_data\ETFs\etf_holdings\{etf_ticker}.xlsx', sheet_name='holdings')
    holdings_info = sector_info[['Name', 'Ticker', 'Weight']].head(10) # pull the top 10 holdings in the sector
    holdings_info.rename(columns={'Name': 'Company'}, inplace=True)

    return holdings_info

