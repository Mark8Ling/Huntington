'''
Generate reports for all ETFs provided.
'''

from data_cleanse import *
from main import create_linear_model 
import os
import glob

def generate_report(PROCESSING, TABLE_CONFIG, etfs):
    tables = {}
    for etf in etfs:
        osl, anova, valid_lag = create_linear_model(PROCESSING, TABLE_CONFIG, etf)
        print(type(osl))
        print(type(anova))
        tables[etf] = [osl, anova, valid_lag]
        # need to save graphs, ols, anova tables, lag applied, all that good stuff
    return tables


def export_html_report(tables, ETF_METADATA, output_path="report.html"):
    html_parts = []

    html_parts.append("""
    <html>
    <head>
        <title>ETF Macro Regression Report</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 40px;
                font-size: 14px;
            }

            h1 {
                font-size: 22px;
            }

            h2 {
                font-size: 18px;
                margin-top: 40px;
            }

            h3 {
                font-size: 15px;
                margin-bottom: 5px;
            }

            table {
                border-collapse: collapse;
                margin-bottom: 30px;
                font-size: 12px;
                width: auto;
                max-width: 900px;
            }

            th, td {
                border: 1px solid #ccc;
                padding: 4px 6px;
                text-align: right;
                white-space: nowrap;
            }

            th {
                background-color: #f2f2f2;
                font-weight: 600;
            }

            hr {
                margin: 50px 0;
            }

            .container {
                max-width: 1000px;
            }
        </style>
    </head>
    <body>
        <div class="container">
        <h1>ETF Macro Regression Report</h1>
    """)

    for etf, (ols_summary, anova_df, valid_lag) in tables.items():
        etf_name = os.path.basename(etf).replace(".csv", "")
        etf_display_name = etf_name.replace("_monthly", "")

        print(f"Processing report for {etf_name}...")
        html_parts.append(f"<hr><h2>{etf_display_name}</h2>")

        # display metadata about ETF from ETF_METADATA
        metadata = ETF_METADATA.get(etf_display_name, {})
        html_parts.append(f"<p><strong>Name:</strong> {metadata.get('name', 'N/A')}</p>")
        html_parts.append(f"<p><strong>Issuer:</strong> {metadata.get('issuer', 'N/A')}</p>")
        html_parts.append(f"<p><strong>Official Page:</strong> <a href='{metadata.get('url', '#')}' target='_blank'>{metadata.get('url', '#')}</a></p>")
        html_parts.append("<h3>Top Holdings</h3>")
        html_parts.append("<ul>")
        for company, weight in metadata.get("holdings", []):
            html_parts.append(f"<li>{company} {weight}</li>")
        html_parts.append("</ul>")

        # Add regression plot image
        image_path = f"reports/images/{etf_name}_results.png"

        html_parts.append("<h3>Train/Test Regression Plot</h3>")
        html_parts.append(
            f'<img src="{image_path}" '
            f'style="max-width:100%; height:auto; margin-bottom:30px; border:1px solid #ccc;" />'
        )

        # OLS Summary
        html_parts.append("<h3>OLS Regression Results</h3>")
        html_parts.append(ols_summary.as_html())

        # ANOVA Table
        html_parts.append("<h3>ANOVA Table</h3>")
        html_parts.append(
            anova_df.to_html(
                float_format="%.4f",
                border=0,
                classes="compact-table"
            )
        )
        # Valid Lags Applied
        html_parts.append("<h3>Valid Lags Applied</h3>")
        if valid_lag:
            html_parts.append("<ul>")
            for col, lag, stability in valid_lag:
                html_parts.append(f"<li>{col}: Lag {lag} (Stability: {stability:.2f})</li>")
            html_parts.append("</ul>")

    html_parts.append("""
        </div>
    </body>
    </html>
    """)

    with open(output_path, "w") as f:
        f.write("".join(html_parts))

    print(f"HTML report saved to {output_path}")

if __name__ == "__main__":
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
        "PCEPI": { 
            "path": "data/raw_data/PCEPI.csv", 
            "pipeline": ["read", "log_diff"], 
            "shift": 0 },
        "UNRATE": { 
            "path": "data/raw_data/UNRATE.csv", 
            "pipeline": ["read", "log_diff"], 
            "shift": 0 },
        "FEDFUNDS": { 
            "path": "data/raw_data/FEDFUNDS.csv", 
            "pipeline": ["read", "log_diff"], 
            "shift": 0 }   
        }

    ETF_METADATA = {
    "XLE": {
        "name": "Energy Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-energy-select-sector-spdr-etf-xle",
        "holdings": [
            ("Exxon Mobil", "24%"),
            ("Chevron", "17%")
        ]
    },
    "XLU": {
        "name": "Utilities Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-utilities-select-sector-spdr-etf-xlu",
        "holdings": [
            ("NextEra Energy", "13.5%"),
            ("Southern Co", "7.23%")
        ]
    },
    "XLF": {
        "name": "Financial Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-financial-select-sector-spdr-etf-xlf",
        "holdings": [
            ("Berkshire Hathaway", "12.20%"),
            ("JPMorgan", "11.12%")
        ]
    },
    "XLK": {
        "name": "Technology Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-technology-select-sector-spdr-etf-xlk",
        "holdings": [
            ("NVIDIA", "15.7%"),
            ("Apple", "13.4%")
        ]
    },
    "XLV": {
        "name": "Health Care Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-health-care-select-sector-spdr-etf-xlv",
        "holdings": [
            ("Eli Lilly", "14.3%"),
            ("Johnson & Johnson", "10.35%")
        ]
    },
    "XLY": {
        "name": "Consumer Discretionary Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-consumer-discretionary-select-sector-spdr-etf-xly",
        "holdings": [
            ("Amazon", "21%"),
            ("Tesla", "19.6%")
        ]
    },
    "XLP": {
        "name": "Consumer Staples Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-consumer-staples-select-sector-spdr-etf-xlp",
        "holdings": [
            ("Walmart", "11.3%"),
            ("Costco", "9%")
        ]
    },
    "XLI": {
        "name": "Industrial Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-industrial-select-sector-spdr-etf-xli",
        "holdings": [
            ("General Electric", "6.72%"),
            ("Caterpillar", "6.66%")
        ]
    },
    "XLB": {
        "name": "Materials Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-materials-select-sector-spdr-etf-xlb",
        "holdings": [
            ("Linde", "14%"),
            ("Newmont", "8%")
        ]
    },
    "XLRE": {
        "name": "Real Estate Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-real-estate-select-sector-spdr-etf-xlre",
        "holdings": [
            ("Welltower", "10.46%"),
            ("Prologis", "9.52%")
        ]
    },
    "XLC": {
        "name": "Communication Services Select Sector SPDR Fund",
        "issuer": "State Street Global Advisors",
        "url": "https://www.ssga.com/us/en/intermediary/etfs/state-street-communication-services-select-sector-spdr-etf-xlc",
        "holdings": [
            ("Meta Platforms", "19.84%"),
            ("Alphabet Class A", "10.78%"),
            ("Alphabet Class C", "8.62%"),
            ("Verizon", ""),
            ("AT&T", ""),
            ("T-Mobile", "")
        ]
    }
}
    
    etf_folder = "data/raw_data/ETFs"
    etfs = glob.glob(os.path.join(etf_folder, "*.csv"))
    tables = generate_report(PROCESSING, TABLE_CONFIG, etfs)
    export_html_report(tables, ETF_METADATA)
    # Everything saves as expected, put into HTML format