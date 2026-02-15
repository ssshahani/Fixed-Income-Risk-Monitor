# Fixed Income Risk Monitor

A Python/Streamlit dashboard for monitoring US Treasury yield curve dynamics, computing duration risk metrics from first principles, and tracking ICE BofA credit spreads — all powered by live FRED data.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-red)
![Data](https://img.shields.io/badge/Data-FRED%20API-green)

## What It Does

| Tab | Description |
|-----|-------------|
| **Yield Curve** | Live Treasury CMT curve with historical overlays and daily changes |
| **Spreads & Regime** | 2s10s, 5s30s, 3m10y, butterfly spreads with inversion regime detection |
| **DV01 & Duration** | DV01, Modified Duration (%), and Convexity computed from first principles |
| **Credit Monitor** | ICE BofA IG/HY/BBB OAS spreads, HY/IG compression ratio |
| **Scenario Lab** | Interactive parallel shift P&L with convexity adjustment |
| **Methodology** | Step-by-step derivation of every calculation with worked examples |

## Data Sources

All data is fetched live from the **FRED API** (Federal Reserve Bank of St. Louis). No sample or synthetic data.

- **Treasury Yields:** US Treasury H.15 Constant Maturity series (DGS1MO through DGS30)
- **Credit Spreads:** ICE BofA Index OAS (BAMLC0A0CM, BAMLH0A0HYM2, BAMLC0A4CBBB, etc.)
- **Fed Funds:** Effective Federal Funds Rate (DFF)

## Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/fi-risk-monitor.git
cd fi-risk-monitor

# Install dependencies
pip install -r requirements.txt

# Add your FRED API key (free at https://fred.stlouisfed.org/docs/api/api_key.html)
cp .streamlit/secrets.toml.example .streamlit/secrets.toml
# Edit .streamlit/secrets.toml and paste your key

# Run
streamlit run app.py
```

## Deploy to Streamlit Community Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the main file
4. Add `FRED_API_KEY` in the app's Secrets settings
5. Deploy — you'll get a public URL for your resume

## Key Technical Concepts Demonstrated

**Fixed Income Math (from first principles):**
- Bond pricing via discounted cash flow
- DV01 via central-difference numerical differentiation
- Modified Duration derived from DV01 (expressed as %)
- Convexity via second-order numerical differentiation
- Full price approximation: dP/P = -ModDur * dy + 0.5 * Convexity * dy^2

**Market Analysis:**
- Post-inversion yield curve regime classification
- Bear steepener dynamics (front end vs long end behavior)
- Credit spread compression analysis (HY/IG ratio, BBB-IG differential)
- Scenario analysis with asymmetric convexity payoff

**Python / Data Engineering:**
- Live API integration (FRED via fredapi)
- Pandas time series manipulation
- Plotly interactive visualizations
- Streamlit caching for performance
- Clean project structure for deployment

## Project Structure

```
fi-risk-monitor/
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md
└── .streamlit/
    ├── config.toml                 # Theme configuration
    └── secrets.toml.example        # API key template
```

## License

MIT
