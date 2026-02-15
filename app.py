"""
Fixed Income Risk Monitor
=========================
Post-Inversion Yield Curve Analysis | Duration Risk | Credit Spreads

A Python/Streamlit dashboard that pulls live data from FRED
(Federal Reserve Economic Data) and computes DV01, Modified Duration,
Convexity, curve spreads, and credit analytics.

Data Sources:
  - US Treasury Constant Maturity Rates (H.15 Release via FRED)
  - ICE BofA Credit Index OAS (via FRED)
  - Federal Funds Effective Rate (via FRED)

Setup:
  1. Get a free FRED API key: https://fred.stlouisfed.org/docs/api/api_key.html
  2. Set it in .streamlit/secrets.toml or as env var FRED_API_KEY
  3. Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from fredapi import Fred
import os

# =====================================================================
# PAGE CONFIG
# =====================================================================
st.set_page_config(
    page_title="Fixed Income Risk Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Dark theme styling — comprehensive override for all Streamlit elements
st.markdown("""
<style>
    /* ── Base ── */
    .stApp { background-color: #0a0f1a; color: #e2e8f0; }

    /* ── All text elements ── */
    .stApp p, .stApp span, .stApp li, .stApp td, .stApp th,
    .stApp label, .stApp div, .stApp caption {
        color: #cbd5e1 !important;
    }

    /* ── Headings ── */
    h1, h2, h3, h4, h5, h6,
    .stApp h1, .stApp h2, .stApp h3, .stApp h4 {
        color: #f1f5f9 !important;
    }

    /* ── Markdown body text ── */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    [data-testid="stMarkdownContainer"],
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] td,
    [data-testid="stMarkdownContainer"] th {
        color: #cbd5e1 !important;
    }

    /* ── Strong / bold ── */
    .stApp strong, .stApp b,
    [data-testid="stMarkdownContainer"] strong {
        color: #f1f5f9 !important;
    }

    /* ── Captions / small text ── */
    .stCaption, [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] p {
        color: #64748b !important;
    }

    /* ── Info / Warning / Error boxes ── */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    .stAlert p {
        color: #e2e8f0 !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] { gap: 4px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #111827;
        border: 1px solid #1e293b;
        border-radius: 6px;
        color: #94a3b8 !important;
        padding: 8px 16px;
    }
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        color: #94a3b8 !important;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
        border-color: #2563eb !important;
    }
    .stTabs [aria-selected="true"] p,
    .stTabs [aria-selected="true"] span {
        color: white !important;
    }
    /* Tab panel content area */
    .stTabs [data-baseweb="tab-panel"],
    .stTabs [data-baseweb="tab-panel"] p,
    .stTabs [data-baseweb="tab-panel"] li,
    .stTabs [data-baseweb="tab-panel"] span {
        color: #cbd5e1 !important;
    }

    /* ── Dataframes / Tables ── */
    .stDataFrame, [data-testid="stDataFrame"] {
        color: #cbd5e1 !important;
    }
    [data-testid="stDataFrame"] td,
    [data-testid="stDataFrame"] th,
    .dvn-scroller td, .dvn-scroller th {
        color: #cbd5e1 !important;
        background-color: #111827 !important;
        border-color: #1e293b !important;
    }
    [data-testid="stDataFrame"] th,
    .dvn-scroller th {
        color: #94a3b8 !important;
        background-color: #0f1520 !important;
    }

    /* Glide data grid (Streamlit's dataframe renderer) */
    .gdg-header, .gdg-cell {
        color: #cbd5e1 !important;
    }

    /* ── Markdown tables ── */
    .stApp table { border-collapse: collapse; width: 100%; }
    .stApp table th {
        color: #94a3b8 !important;
        background-color: #0f1520 !important;
        border: 1px solid #1e293b !important;
        padding: 8px 12px !important;
        text-align: left;
    }
    .stApp table td {
        color: #cbd5e1 !important;
        background-color: #111827 !important;
        border: 1px solid #1e293b !important;
        padding: 8px 12px !important;
    }

    /* ── Code blocks ── */
    .stCode, .stApp pre, .stApp code,
    [data-testid="stCode"] pre,
    [data-testid="stCode"] code {
        color: #e2e8f0 !important;
        background-color: #111827 !important;
        border: 1px solid #1e293b;
    }

    /* ── LaTeX / math ── */
    .katex, .katex * { color: #e2e8f0 !important; }

    /* ── Multiselect / input widgets ── */
    [data-baseweb="select"] { background-color: #111827 !important; }
    [data-baseweb="select"] span,
    [data-baseweb="select"] div {
        color: #cbd5e1 !important;
    }
    [data-baseweb="tag"] {
        background-color: #1e3a5f !important;
        color: #93c5fd !important;
    }

    /* ── Sliders ── */
    .stSlider label, .stSlider p,
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] p,
    [data-testid="stSlider"] span {
        color: #cbd5e1 !important;
    }

    /* ── Metric cards (custom) ── */
    .metric-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
    }
    .metric-label { color: #64748b !important; font-size: 11px; letter-spacing: 0.08em; }
    .metric-value { font-size: 26px; font-weight: 700; margin-top: 4px; }
    .metric-sub { color: #64748b !important; font-size: 11px; margin-top: 2px; }

    /* ── Expander ── */
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] p {
        color: #cbd5e1 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] { background-color: #0f1520; }
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }

    /* ── Input text field ── */
    .stTextInput input {
        color: #e2e8f0 !important;
        background-color: #111827 !important;
        border-color: #1e293b !important;
    }

    /* ── Horizontal rule ── */
    hr { border-color: #1e293b !important; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# FRED DATA FETCHING
# =====================================================================

# Treasury yield series (Constant Maturity)
YIELD_SERIES = {
    "1M": "DGS1MO", "3M": "DGS3MO", "6M": "DGS6MO",
    "1Y": "DGS1", "2Y": "DGS2", "3Y": "DGS3",
    "5Y": "DGS5", "7Y": "DGS7", "10Y": "DGS10",
    "20Y": "DGS20", "30Y": "DGS30",
}

# ICE BofA Credit OAS series
CREDIT_SERIES = {
    "AAA OAS": "BAMLC0A1CAAA",
    "IG OAS": "BAMLC0A0CM",
    "A OAS": "BAMLC0A3CA",
    "BBB OAS": "BAMLC0A4CBBB",
    "HY OAS": "BAMLH0A0HYM2",
    "BB OAS": "BAMLH0A1HYBB",
    "B OAS": "BAMLH0A2HYB",
}

# Fed Funds
FED_FUNDS_SERIES = "DFF"

# Tenor in years (for duration calcs)
TENOR_YEARS = {
    "1M": 1/12, "3M": 0.25, "6M": 0.5, "1Y": 1.0, "2Y": 2.0,
    "3Y": 3.0, "5Y": 5.0, "7Y": 7.0, "10Y": 10.0, "20Y": 20.0, "30Y": 30.0,
}


def get_fred_api_key():
    """Retrieve FRED API key from Streamlit secrets or environment."""
    try:
        return st.secrets["FRED_API_KEY"]
    except Exception:
        key = os.environ.get("FRED_API_KEY", "")
        if key:
            return key
    return None


@st.cache_data(ttl=3600, show_spinner="Fetching Treasury yield data from FRED...")
def fetch_yields(api_key, start_date, end_date):
    """Fetch US Treasury CMT rates from FRED."""
    fred = Fred(api_key=api_key)
    frames = {}
    for label, sid in YIELD_SERIES.items():
        try:
            s = fred.get_series(sid, start_date, end_date)
            frames[label] = s
        except Exception:
            pass
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all").ffill()
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching credit spread data from FRED...")
def fetch_credit(api_key, start_date, end_date):
    """Fetch ICE BofA credit OAS indices from FRED."""
    fred = Fred(api_key=api_key)
    frames = {}
    for label, sid in CREDIT_SERIES.items():
        try:
            s = fred.get_series(sid, start_date, end_date)
            frames[label] = s
        except Exception:
            pass
    df = pd.DataFrame(frames)
    df.index = pd.to_datetime(df.index)
    df = df.dropna(how="all").ffill()
    return df


@st.cache_data(ttl=3600, show_spinner="Fetching Fed Funds rate...")
def fetch_fed_funds(api_key, start_date, end_date):
    """Fetch effective Fed Funds rate."""
    fred = Fred(api_key=api_key)
    try:
        s = fred.get_series(FED_FUNDS_SERIES, start_date, end_date)
        return s.dropna()
    except Exception:
        return pd.Series(dtype=float)


# =====================================================================
# FIXED INCOME MATH
# =====================================================================

def bond_price(ytm_pct, maturity_years, coupon_rate_pct=None, freq=2, face=100):
    """
    Price a fixed-coupon bond using discounted cash flow.

    Parameters
    ----------
    ytm_pct : float         Yield to maturity (e.g. 4.5 for 4.5%)
    maturity_years : float  Time to maturity in years
    coupon_rate_pct : float  Annual coupon rate in % (defaults to ytm for par bond)
    freq : int              Coupon frequency (2 = semi-annual)
    face : float            Face value

    Returns
    -------
    float : Clean price
    """
    if coupon_rate_pct is None:
        coupon_rate_pct = ytm_pct  # par bond assumption

    y = ytm_pct / 100
    c_rate = coupon_rate_pct / 100
    n = max(1, round(maturity_years * freq))
    coupon = (c_rate / freq) * face
    r = y / freq

    if abs(r) < 1e-12:
        return coupon * n + face

    pv_coupons = coupon * (1 - (1 + r) ** (-n)) / r
    pv_face = face / (1 + r) ** n
    return pv_coupons + pv_face


def compute_dv01(ytm_pct, maturity_years, freq=2, face=100):
    """
    DV01: Dollar Value of a Basis Point.

    Method: Numerical differentiation (central difference)
      DV01 = [Price(y - 1bp) - Price(y + 1bp)] / 2

    For a $100 par bond at 4.09% yield, 10Y maturity:
      DV01 tells you how many dollars the price moves per 1bp yield change.

    Returns: dollars per $100 face value
    """
    bp = 0.01  # 1 basis point in percent
    p_down = bond_price(ytm_pct - bp, maturity_years, ytm_pct, freq, face)
    p_up = bond_price(ytm_pct + bp, maturity_years, ytm_pct, freq, face)
    return (p_down - p_up) / 2


def compute_modified_duration(ytm_pct, maturity_years, freq=2, face=100):
    """
    Modified Duration: Percentage price sensitivity per 1% yield change.

    Derived from DV01:
      ModDur = DV01 * 10,000 / Price

    For a par bond (Price = 100):
      ModDur = DV01 * 100

    Interpretation: If ModDur = 8.12%, then a +100bp yield move
    causes approximately -8.12% price decline.

    Returns: percentage (e.g. 8.12 means 8.12%)
    """
    dv01 = compute_dv01(ytm_pct, maturity_years, freq, face)
    price = bond_price(ytm_pct, maturity_years, ytm_pct, freq, face)
    return (dv01 / price) * 10000  # in %


def compute_convexity(ytm_pct, maturity_years, freq=2, face=100):
    """
    Convexity: Curvature of the price-yield relationship.

    Method: Second-order numerical differentiation
      Convexity = [P(y-1bp) + P(y+1bp) - 2*P(y)] / [P(y) * (1bp)^2]

    Used to improve the Duration approximation:
      dP/P ~ -ModDur * dy + 0.5 * Convexity * dy^2

    Returns: convexity in years-squared
    """
    bp = 0.0001  # 1bp as decimal
    y = ytm_pct / 100
    p_mid = bond_price(ytm_pct, maturity_years, ytm_pct, freq, face)
    p_down = bond_price(ytm_pct - 0.01, maturity_years, ytm_pct, freq, face)
    p_up = bond_price(ytm_pct + 0.01, maturity_years, ytm_pct, freq, face)
    return (p_down + p_up - 2 * p_mid) / (p_mid * bp * bp)


def compute_curve_spreads(yields_df):
    """Compute key yield curve spreads in basis points."""
    spreads = pd.DataFrame(index=yields_df.index)

    if "10Y" in yields_df and "2Y" in yields_df:
        spreads["2s10s (bp)"] = (yields_df["10Y"] - yields_df["2Y"]) * 100
    if "30Y" in yields_df and "5Y" in yields_df:
        spreads["5s30s (bp)"] = (yields_df["30Y"] - yields_df["5Y"]) * 100
    if "10Y" in yields_df and "3M" in yields_df:
        spreads["3m10y (bp)"] = (yields_df["10Y"] - yields_df["3M"]) * 100
    if "5Y" in yields_df and "2Y" in yields_df and "10Y" in yields_df:
        spreads["2s5s10s Fly (bp)"] = (
            2 * yields_df["5Y"] - yields_df["2Y"] - yields_df["10Y"]
        ) * 100

    return spreads.dropna(how="all")


# =====================================================================
# PLOTLY THEME
# =====================================================================

PLOT_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#0a0f1a",
    plot_bgcolor="#0f1520",
    font=dict(family="JetBrains Mono, SF Mono, monospace", color="#94a3b8", size=11),
    margin=dict(l=50, r=30, t=40, b=40),
    xaxis=dict(gridcolor="#1e293b", zeroline=False),
    yaxis=dict(gridcolor="#1e293b", zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
)

COLORS = {
    "blue": "#2d8cf0", "red": "#ff4757", "green": "#00c48c",
    "amber": "#ffa502", "purple": "#7c5cfc", "cyan": "#17c0eb",
    "pink": "#e84393",
}


# =====================================================================
# STREAMLIT APP
# =====================================================================

def main():
    # --- Header ---
    col_h1, col_h2 = st.columns([3, 2])
    with col_h1:
        st.markdown("# Fixed Income Risk Monitor")
        st.caption(
            "Post-Inversion Yield Curve Analysis  |  DV01 & Duration Risk  |  "
            "ICE BofA Credit Spreads  |  Live FRED Data"
        )

    # --- API Key ---
    api_key = get_fred_api_key()
    if not api_key:
        st.error(
            "**FRED API key required.** "
            "Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html "
            "then add it to `.streamlit/secrets.toml` as `FRED_API_KEY = \"your_key\"` "
            "or set the environment variable `FRED_API_KEY`."
        )
        api_key = st.text_input("Enter FRED API Key:", type="password")
        if not api_key:
            st.stop()

    # --- Fetch Data ---
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_3y = (datetime.today() - timedelta(days=365 * 4)).strftime("%Y-%m-%d")

    yields_df = fetch_yields(api_key, start_3y, end_date)
    credit_df = fetch_credit(api_key, start_3y, end_date)
    fed_funds = fetch_fed_funds(api_key, start_3y, end_date)

    if yields_df.empty:
        st.error("Failed to fetch yield data. Check your API key.")
        st.stop()

    latest = yields_df.dropna(how="all").iloc[-1]
    latest_date = yields_df.dropna(how="all").index[-1]
    prev = yields_df.dropna(how="all").iloc[-2] if len(yields_df) > 1 else latest

    # --- Top Metrics Bar ---
    spreads = compute_curve_spreads(yields_df)
    latest_spreads = spreads.iloc[-1] if not spreads.empty else pd.Series()

    metric_cols = st.columns(7)
    metrics_data = [
        ("3M UST", f"{latest.get('3M', 0):.2f}%", ""),
        ("10Y UST", f"{latest.get('10Y', 0):.2f}%", ""),
        ("30Y UST", f"{latest.get('30Y', 0):.2f}%", ""),
        ("2s10s", f"{latest_spreads.get('2s10s (bp)', 0):+.0f}bp", ""),
        ("5s30s", f"{latest_spreads.get('5s30s (bp)', 0):+.0f}bp", ""),
    ]
    if not credit_df.empty:
        lc = credit_df.dropna(how="all").iloc[-1]
        metrics_data.append(("IG OAS", f"{lc.get('IG OAS', 0)*100:.0f}bp", ""))
        metrics_data.append(("HY OAS", f"{lc.get('HY OAS', 0)*100:.0f}bp", ""))

    for i, (label, val, sub) in enumerate(metrics_data[:7]):
        with metric_cols[i]:
            color = "#2d8cf0" if i < 3 else "#00c48c" if i < 5 else "#ffa502" if i == 5 else "#ff4757"
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">{label}</div>'
                f'<div class="metric-value" style="color:{color}">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="text-align:right; color:#4a5b72; font-size:11px; margin-top:4px;">'
        f'Data as of {latest_date.strftime("%B %d, %Y")} | Source: FRED / US Treasury H.15'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ================================================================
    # TABS
    # ================================================================
    tabs = st.tabs([
        "Yield Curve",
        "Spreads & Regime",
        "DV01 & Duration",
        "Credit Monitor",
        "Scenario Lab",
        "Methodology",
    ])

    # ----------------------------------------------------------------
    # TAB 0: YIELD CURVE
    # ----------------------------------------------------------------
    with tabs[0]:
        st.subheader("US Treasury Par Yield Curve - Constant Maturity Rates")

        # Current curve
        tenors = [t for t in TENOR_YEARS if t in latest.index and pd.notna(latest[t])]
        years = [TENOR_YEARS[t] for t in tenors]
        current_yields = [latest[t] for t in tenors]

        # Comparison dates
        st.markdown("**Select comparison dates:**")
        compare_options = {
            "1 week ago": 5,
            "1 month ago": 21,
            "3 months ago": 63,
            "6 months ago": 126,
            "1 year ago": 252,
            "2 years ago": 504,
        }
        selected_comparisons = st.multiselect(
            "Compare against:",
            list(compare_options.keys()),
            default=["1 year ago", "2 years ago"],
            label_visibility="collapsed",
        )

        fig_curve = go.Figure()
        fig_curve.add_trace(go.Scatter(
            x=tenors, y=current_yields, mode="lines+markers",
            name=f"Current ({latest_date.strftime('%b %d, %Y')})",
            line=dict(color=COLORS["blue"], width=3),
            marker=dict(size=8),
        ))

        comp_colors = [COLORS["red"], COLORS["amber"], COLORS["green"],
                       COLORS["purple"], COLORS["cyan"], COLORS["pink"]]
        for idx, label in enumerate(selected_comparisons):
            offset = compare_options[label]
            if offset < len(yields_df):
                comp_row = yields_df.iloc[-(offset + 1)]
                comp_date = yields_df.index[-(offset + 1)]
                comp_yields = [comp_row.get(t, np.nan) for t in tenors]
                fig_curve.add_trace(go.Scatter(
                    x=tenors, y=comp_yields, mode="lines+markers",
                    name=f"{label} ({comp_date.strftime('%b %d, %Y')})",
                    line=dict(color=comp_colors[idx % len(comp_colors)], width=1.5, dash="dash"),
                    marker=dict(size=4),
                ))

        fig_curve.update_layout(
            **PLOT_LAYOUT,
            title="Treasury Yield Curve",
            yaxis_title="Yield (%)",
            xaxis_title="Maturity",
            height=420,
        )
        st.plotly_chart(fig_curve, use_container_width=True)

        # Current curve table with daily changes
        st.markdown("**Current Curve Detail**")
        table_data = []
        for t in tenors:
            curr = latest[t]
            prev_val = prev.get(t, np.nan)
            chg_1d = (curr - prev_val) * 100 if pd.notna(prev_val) else 0

            # 1Y ago comparison
            if len(yields_df) > 252:
                y1_ago = yields_df.iloc[-253].get(t, np.nan)
                chg_1y = (curr - y1_ago) * 100 if pd.notna(y1_ago) else np.nan
            else:
                chg_1y = np.nan

            table_data.append({
                "Tenor": t,
                "Yield (%)": f"{curr:.2f}",
                "1d Change (bp)": f"{chg_1d:+.0f}",
                "vs 1Y Ago (bp)": f"{chg_1y:+.0f}" if pd.notna(chg_1y) else "-",
            })

        st.dataframe(
            pd.DataFrame(table_data).set_index("Tenor"),
            use_container_width=True,
        )
        st.caption("Source: US Treasury Dept - H.15 Selected Interest Rates via FRED")

    # ----------------------------------------------------------------
    # TAB 1: SPREADS & REGIME
    # ----------------------------------------------------------------
    with tabs[1]:
        st.subheader("Yield Curve Spreads & Regime Analysis")

        if not spreads.empty:
            # Spread summary cards
            sp_cols = st.columns(4)
            spread_items = [
                ("2s10s", "2s10s (bp)", "10Y minus 2Y"),
                ("5s30s", "5s30s (bp)", "30Y minus 5Y"),
                ("3m10y", "3m10y (bp)", "10Y minus 3M"),
                ("Butterfly", "2s5s10s Fly (bp)", "2x5Y - 2Y - 10Y"),
            ]
            for i, (label, col_name, desc) in enumerate(spread_items):
                with sp_cols[i]:
                    val = latest_spreads.get(col_name, 0)
                    color = "#00c48c" if val > 0 else "#ff4757" if val < 0 else "#94a3b8"
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{label.upper()}</div>'
                        f'<div class="metric-value" style="color:{color}">{val:+.0f}bp</div>'
                        f'<div class="metric-sub">{desc}</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # 2s10s time series
            fig_spread = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                       vertical_spacing=0.08,
                                       subplot_titles=("2s10s Treasury Spread (bp)",
                                                       "5s30s Treasury Spread (bp)"))

            if "2s10s (bp)" in spreads:
                s2s10 = spreads["2s10s (bp)"].dropna()
                fig_spread.add_trace(
                    go.Scatter(x=s2s10.index, y=s2s10, mode="lines",
                               line=dict(color=COLORS["blue"], width=2), name="2s10s"),
                    row=1, col=1,
                )
                fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["red"],
                                     line_width=1, row=1, col=1)
                # Fill inversion
                fig_spread.add_trace(
                    go.Scatter(x=s2s10.index, y=s2s10.clip(upper=0),
                               fill="tozeroy", fillcolor="rgba(255,71,87,0.15)",
                               line=dict(width=0), showlegend=False, name="Inverted"),
                    row=1, col=1,
                )

            if "5s30s (bp)" in spreads:
                s5s30 = spreads["5s30s (bp)"].dropna()
                fig_spread.add_trace(
                    go.Scatter(x=s5s30.index, y=s5s30, mode="lines",
                               line=dict(color=COLORS["amber"], width=2), name="5s30s"),
                    row=2, col=1,
                )
                fig_spread.add_hline(y=0, line_dash="dash", line_color=COLORS["red"],
                                     line_width=1, row=2, col=1)

            fig_spread.update_layout(**PLOT_LAYOUT, height=550, showlegend=True)
            fig_spread.update_yaxes(title_text="bp", row=1, col=1)
            fig_spread.update_yaxes(title_text="bp", row=2, col=1)
            st.plotly_chart(fig_spread, use_container_width=True)

            # Regime classification
            if "2s10s (bp)" in spreads:
                s = spreads["2s10s (bp)"].dropna()
                regime = pd.cut(
                    s,
                    bins=[-np.inf, -50, 0, 50, 150, np.inf],
                    labels=["Deep Inversion", "Inverted", "Flat", "Normal", "Steep"],
                )
                current_regime = regime.iloc[-1]
                days_inverted = (s < 0).sum()
                st.info(
                    f"**Current Regime:** {current_regime}  |  "
                    f"**Days inverted** in this window: {days_inverted}  |  "
                    f"**Current 2s10s:** {s.iloc[-1]:+.0f}bp"
                )

            # Market analysis
            y2_curr = latest.get("2Y", np.nan)
            y10_curr = latest.get("10Y", np.nan)
            y30_curr = latest.get("30Y", np.nan)

            # Look back ~2 years for peak inversion comparison
            if len(yields_df) > 400:
                s2s10_hist = (yields_df["10Y"] - yields_df["2Y"]).dropna() * 100
                min_idx = s2s10_hist.idxmin()
                min_val = s2s10_hist.min()
                y2_at_min = yields_df.loc[min_idx, "2Y"]
                y10_at_min = yields_df.loc[min_idx, "10Y"]
                chg_2y = (y2_curr - y2_at_min) * 100
                chg_10y = (y10_curr - y10_at_min) * 100

                st.markdown("---")
                st.markdown("#### Bull Steepener Analysis")
                st.markdown(
                    f"The 2s10s spread hit **{min_val:+.0f}bp** on "
                    f"{min_idx.strftime('%b %d, %Y')} - the deepest inversion since the early 1980s. "
                    f"Since then, normalization has been a **bull steepener**: the front end rallied "
                    f"aggressively as the Fed cut rates while the long end held steady or backed up."
                )

                col_a, col_b = st.columns(2)
                with col_a:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">2Y CHANGE SINCE PEAK INV.</div>'
                        f'<div class="metric-value" style="color:{"#00c48c" if chg_2y < 0 else "#ff4757"}">'
                        f'{chg_2y:+.0f}bp</div>'
                        f'<div class="metric-sub">{y2_at_min:.2f}% -> {y2_curr:.2f}% (front end rallied)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                with col_b:
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">10Y CHANGE SINCE PEAK INV.</div>'
                        f'<div class="metric-value" style="color:{"#00c48c" if chg_10y < 0 else "#ff4757"}">'
                        f'{chg_10y:+.0f}bp</div>'
                        f'<div class="metric-sub">{y10_at_min:.2f}% -> {y10_curr:.2f}% (long end backed up)</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                st.markdown(
                    f"In a **bull steepener**, short-term rates fall faster than long-term rates. "
                    f"Here the 2Y dropped **{abs(chg_2y):.0f}bp** while the 10Y moved just "
                    f"**{chg_10y:+.0f}bp** - the steepening is overwhelmingly driven by the "
                    f"front-end rally on Fed rate cuts, not a long-end selloff. "
                    f"The 5s30s spread at "
                    f"**{latest_spreads.get('5s30s (bp)', 0):+.0f}bp** reflects additional "
                    f"long-end steepness from term premium and Treasury supply pressure."
                )

        st.caption("Source: Computed from FRED Treasury CMT series (DGS2, DGS10, DGS30, DGS5, DGS3MO)")

    # ----------------------------------------------------------------
    # TAB 2: DV01 & DURATION
    # ----------------------------------------------------------------
    with tabs[2]:
        st.subheader("Duration & DV01 Risk Profile - Current Curve")

        risk_data = []
        for tenor in tenors:
            yld = latest[tenor]
            yrs = TENOR_YEARS[tenor]
            if pd.isna(yld) or yrs < 0.25:
                continue
            dv01 = compute_dv01(yld, yrs)
            mod_dur = compute_modified_duration(yld, yrs)
            convex = compute_convexity(yld, yrs)

            # Price impact for +/- 50bp
            shift = 0.50  # 50bp in %
            shift_dec = shift / 100
            pnl_up = -mod_dur / 100 * shift + 0.5 * convex * shift_dec ** 2 * 100
            pnl_down = mod_dur / 100 * shift + 0.5 * convex * shift_dec ** 2 * 100

            risk_data.append({
                "Tenor": tenor,
                "Maturity (yr)": yrs,
                "Yield (%)": round(yld, 3),
                "DV01 ($)": round(dv01, 4),
                "Mod Duration (%)": round(mod_dur, 2),
                "Convexity": round(convex, 1),
                "Price Chg +50bp (%)": round(pnl_up, 2),
                "Price Chg -50bp (%)": round(pnl_down, 2),
            })

        risk_df = pd.DataFrame(risk_data)

        # DV01 bar chart
        fig_dv01 = go.Figure()
        fig_dv01.add_trace(go.Bar(
            x=risk_df["Tenor"], y=risk_df["DV01 ($)"],
            marker_color=[f"hsl({210 + i*12}, 65%, {50 + i*2}%)" for i in range(len(risk_df))],
            text=[f"${v:.3f}" for v in risk_df["DV01 ($)"]],
            textposition="outside",
            textfont=dict(size=9),
        ))
        fig_dv01.update_layout(
            **PLOT_LAYOUT,
            title="DV01 by Tenor ($ per $100 face per 1bp yield move)",
            yaxis_title="DV01 ($)",
            xaxis_title="Tenor",
            height=350,
        )
        st.plotly_chart(fig_dv01, use_container_width=True)

        # Modified Duration + Yield overlay
        fig_dur = make_subplots(specs=[[{"secondary_y": True}]])
        fig_dur.add_trace(
            go.Bar(x=risk_df["Tenor"], y=risk_df["Mod Duration (%)"],
                   name="Mod Duration (%)", marker_color=COLORS["purple"],
                   opacity=0.6),
            secondary_y=False,
        )
        fig_dur.add_trace(
            go.Scatter(x=risk_df["Tenor"], y=risk_df["Yield (%)"],
                       name="Yield (%)", mode="lines+markers",
                       line=dict(color=COLORS["amber"], width=2),
                       marker=dict(size=6)),
            secondary_y=True,
        )
        fig_dur.update_layout(**PLOT_LAYOUT, title="Modified Duration vs Yield", height=350)
        fig_dur.update_yaxes(title_text="Mod Duration (%)", secondary_y=False)
        fig_dur.update_yaxes(title_text="Yield (%)", secondary_y=True)
        st.plotly_chart(fig_dur, use_container_width=True)

        # Risk table
        st.markdown("**Risk Metrics Table**")
        st.markdown(
            "_Modified Duration (%) = approximate price change for a 100bp yield move. "
            "DV01 = dollar price change per 1bp yield move on $100 face._"
        )
        st.dataframe(
            risk_df.set_index("Tenor"),
            use_container_width=True,
        )

    # ----------------------------------------------------------------
    # TAB 3: CREDIT MONITOR
    # ----------------------------------------------------------------
    with tabs[3]:
        st.subheader("Credit Spread Monitor - ICE BofA Index Suite")

        if credit_df.empty:
            st.warning("Credit spread data not available. Check FRED API access.")
        else:
            lc = credit_df.dropna(how="all").iloc[-1]
            lc_date = credit_df.dropna(how="all").index[-1]

            # Current spread cards
            cr_cols = st.columns(len(credit_df.columns))
            card_colors = [COLORS["cyan"], COLORS["blue"], COLORS["green"],
                           COLORS["amber"], COLORS["red"], COLORS["purple"], COLORS["pink"]]
            for i, col in enumerate(credit_df.columns):
                with cr_cols[i]:
                    val = lc.get(col, 0)
                    val_bp = val * 100 if abs(val) < 10 else val  # handle pct vs bp
                    st.markdown(
                        f'<div class="metric-card">'
                        f'<div class="metric-label">{col}</div>'
                        f'<div class="metric-value" style="color:{card_colors[i % len(card_colors)]}">'
                        f'{val_bp:.0f}<span style="font-size:12px">bp</span></div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            # OAS time series
            fig_credit = go.Figure()
            cr_line_colors = card_colors
            for i, col in enumerate(credit_df.columns):
                series = credit_df[col].dropna()
                vals = series * 100 if series.max() < 10 else series
                fig_credit.add_trace(go.Scatter(
                    x=vals.index, y=vals, mode="lines",
                    name=col,
                    line=dict(color=cr_line_colors[i % len(cr_line_colors)],
                              width=2 if "IG" in col or "HY" in col or "BBB" in col else 1.5,
                              dash=None if "IG" in col or "HY" in col or "BBB" in col else "dash"),
                ))
            fig_credit.update_layout(
                **PLOT_LAYOUT,
                title="ICE BofA Option-Adjusted Spreads (bp)",
                yaxis_title="OAS (bp)", height=420,
            )
            st.plotly_chart(fig_credit, use_container_width=True)

            # HY/IG ratio
            if "HY OAS" in credit_df and "IG OAS" in credit_df:
                ratio = (credit_df["HY OAS"] / credit_df["IG OAS"]).dropna()
                fig_ratio = go.Figure()
                fig_ratio.add_trace(go.Scatter(
                    x=ratio.index, y=ratio, mode="lines",
                    line=dict(color=COLORS["cyan"], width=2), name="HY/IG Ratio",
                ))
                fig_ratio.add_hline(
                    y=ratio.rolling(252, min_periods=60).mean().iloc[-1],
                    line_dash="dash", line_color="#4a5b72",
                    annotation_text="1Y Avg",
                )
                fig_ratio.update_layout(
                    **PLOT_LAYOUT,
                    title="HY/IG Compression Ratio",
                    yaxis_title="Ratio (x)", height=300,
                )
                st.plotly_chart(fig_ratio, use_container_width=True)

            st.caption(
                f"Source: ICE BofA Indices via FRED | "
                f"BAMLC0A0CM (IG), BAMLH0A0HYM2 (HY), BAMLC0A4CBBB (BBB), "
                f"BAMLC0A1CAAA (AAA), BAMLC0A3CA (A) | As of {lc_date.strftime('%b %d, %Y')}"
            )

    # ----------------------------------------------------------------
    # TAB 4: SCENARIO LAB
    # ----------------------------------------------------------------
    with tabs[4]:
        st.subheader("Parallel Shift Scenario Analysis")

        sc1, sc2 = st.columns(2)
        with sc1:
            shift_bp = st.slider("Parallel Shift (bp)", 10, 200, 50, 5)
        with sc2:
            port_size = st.slider("Portfolio Size ($M)", 10, 500, 100, 10)

        notional_per = (port_size * 1e6) / len(risk_data)
        scenario_rows = []
        total_up = 0
        total_down = 0

        for r in risk_data:
            shift_dec = shift_bp / 10000
            mod_dur = r["Mod Duration (%)"]
            convex = r["Convexity"]

            pnl_up = notional_per * (-mod_dur / 100 * shift_dec + 0.5 * convex * shift_dec ** 2)
            pnl_down = notional_per * (mod_dur / 100 * shift_dec + 0.5 * convex * shift_dec ** 2)
            total_up += pnl_up
            total_down += pnl_down

            scenario_rows.append({
                "Tenor": r["Tenor"],
                "Notional ($M)": round(notional_per / 1e6, 1),
                "DV01": f"${r['DV01 ($)']:.4f}",
                "Mod Dur (%)": f"{mod_dur:.2f}",
                f"P&L +{shift_bp}bp ($)": f"{pnl_up:,.0f}",
                f"P&L -{shift_bp}bp ($)": f"{pnl_down:,.0f}",
            })

        # Summary
        s_cols = st.columns(2)
        with s_cols[0]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">RATES +{shift_bp}bp</div>'
                f'<div class="metric-value" style="color:#ff4757">${total_up/1e6:,.2f}M</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with s_cols[1]:
            st.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-label">RATES -{shift_bp}bp</div>'
                f'<div class="metric-value" style="color:#00c48c">+${total_down/1e6:,.2f}M</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

        # Bar chart
        sc_df = pd.DataFrame(scenario_rows)
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Bar(
            x=sc_df["Tenor"],
            y=[float(v.replace(",", "")) for v in sc_df[f"P&L +{shift_bp}bp ($)"]],
            name=f"Rates +{shift_bp}bp", marker_color=COLORS["red"], opacity=0.7,
        ))
        fig_sc.add_trace(go.Bar(
            x=sc_df["Tenor"],
            y=[float(v.replace(",", "")) for v in sc_df[f"P&L -{shift_bp}bp ($)"]],
            name=f"Rates -{shift_bp}bp", marker_color=COLORS["green"], opacity=0.7,
        ))
        fig_sc.update_layout(
            **PLOT_LAYOUT, barmode="group",
            title=f"P&L by Tenor - +/-{shift_bp}bp on ${port_size}M Portfolio",
            yaxis_title="P&L ($)", height=380,
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.dataframe(pd.DataFrame(scenario_rows).set_index("Tenor"), use_container_width=True)
        st.caption(
            "P&L = Notional x (-ModDur x dy + 0.5 x Convexity x dy^2). "
            "Equal-weighted allocation across tenors. "
            "Convexity creates asymmetric payoff: gains from rate drops exceed losses from equal rate rises."
        )

    # ----------------------------------------------------------------
    # TAB 5: METHODOLOGY
    # ----------------------------------------------------------------
    with tabs[5]:
        st.subheader("Calculation Methodology")
        st.markdown(
            "This section explains how each risk metric is computed from first principles. "
            "All calculations assume **par bonds** (coupon rate = yield) with **semi-annual** coupon frequency, "
            "which is the standard convention for US Treasuries."
        )

        # --- Bond Price ---
        st.markdown("---")
        st.markdown("### 1. Bond Price (Discounted Cash Flow)")
        st.markdown("""
The price of a fixed-coupon bond is the present value of all future cash flows:

$$P = \\sum_{t=1}^{n} \\frac{C}{(1 + r)^t} + \\frac{F}{(1 + r)^n}$$

Where:
- **C** = coupon payment per period = (annual coupon rate / frequency) x face value
- **r** = yield per period = annual yield / frequency
- **n** = total number of coupon periods = maturity in years x frequency
- **F** = face value (par = $100)

For a **par bond**, the coupon rate equals the yield, so P = $100 by definition.
This is our starting assumption for computing DV01 and duration.
        """)

        # Worked example
        st.markdown("**Worked Example: 10Y Treasury at 4.09%**")
        ex_yield = latest.get("10Y", 4.09)
        ex_price = bond_price(ex_yield, 10)
        st.code(f"""
Yield     = {ex_yield:.2f}%
Maturity  = 10 years
Frequency = 2 (semi-annual)
Periods   = 10 x 2 = 20
Coupon    = ({ex_yield:.2f}% / 2) x $100 = ${ex_yield/2:.4f}
Price     = ${ex_price:.4f}  (par bond -> ~$100)
""", language="text")

        # --- DV01 ---
        st.markdown("---")
        st.markdown("### 2. DV01 (Dollar Value of a Basis Point)")
        st.markdown("""
DV01 measures the **dollar change in bond price for a 1 basis point (0.01%) change in yield**.
It is computed using **central difference numerical differentiation**:

$$DV01 = \\frac{P(y - 1bp) - P(y + 1bp)}{2}$$

Where P(y) is the bond price at yield y. We shift the yield up and down by exactly 1bp
and take the average price change.

**Why DV01 matters:** It directly tells you P&L exposure. If you hold $10M face of a bond
with DV01 = $0.0810, then a 1bp rate move costs/gains you:

$$P\\&L = \\frac{\\$10{,}000{,}000}{100} \\times \\$0.0810 = \\$8{,}100 \\text{ per basis point}$$
        """)

        ex_dv01 = compute_dv01(ex_yield, 10)
        st.markdown("**Worked Example: 10Y at {:.2f}%**".format(ex_yield))
        p_down = bond_price(ex_yield - 0.01, 10, ex_yield)
        p_up = bond_price(ex_yield + 0.01, 10, ex_yield)
        st.code(f"""
Price at {ex_yield - 0.01:.2f}% = ${p_down:.6f}
Price at {ex_yield + 0.01:.2f}% = ${p_up:.6f}

DV01 = ({p_down:.6f} - {p_up:.6f}) / 2 = ${ex_dv01:.6f}

Interpretation: For every 1bp move in the 10Y yield,
the bond price changes by ${ex_dv01:.4f} per $100 face.
""", language="text")

        # --- Modified Duration ---
        st.markdown("---")
        st.markdown("### 3. Modified Duration (% Price Sensitivity)")
        st.markdown("""
Modified Duration expresses interest rate sensitivity as a **percentage of price**.
It is derived from DV01:

$$\\text{Modified Duration (\\%)} = \\frac{DV01}{Price} \\times 10{,}000$$

**Interpretation:** Modified Duration (%) tells you the approximate **percentage price change
for a 100bp (1%) move in yield**.

For example, if Mod Duration = 8.12%, then:
- Yield rises 100bp -> Price falls ~8.12%
- Yield falls 50bp -> Price rises ~4.06%

This is the **first-order linear approximation** of the price-yield relationship.
        """)

        ex_moddur = compute_modified_duration(ex_yield, 10)
        st.markdown("**Worked Example: 10Y at {:.2f}%**".format(ex_yield))
        st.code(f"""
DV01  = ${ex_dv01:.6f}
Price = ${ex_price:.4f}

Mod Duration = ({ex_dv01:.6f} / {ex_price:.4f}) x 10,000
             = {ex_moddur:.2f}%

Interpretation: A 100bp yield increase causes ~{ex_moddur:.2f}% price decline.
A 50bp yield decrease causes ~{ex_moddur/2:.2f}% price gain.
""", language="text")

        # --- Convexity ---
        st.markdown("---")
        st.markdown("### 4. Convexity (Curvature Correction)")
        st.markdown("""
Duration is a linear approximation. For large yield moves, the actual price-yield
relationship is **curved** (convex). Convexity captures this curvature:

$$\\text{Convexity} = \\frac{P(y - 1bp) + P(y + 1bp) - 2 \\cdot P(y)}{P(y) \\cdot (\\Delta y)^2}$$

The **full price change approximation** including convexity is:

$$\\frac{\\Delta P}{P} \\approx -\\text{ModDur} \\times \\Delta y + \\frac{1}{2} \\times \\text{Convexity} \\times (\\Delta y)^2$$

**Key insight:** The convexity term is always positive (for vanilla bonds), which means:
- When rates **rise**, convexity **reduces** losses vs the Duration-only estimate
- When rates **fall**, convexity **increases** gains vs the Duration-only estimate

This creates an **asymmetric payoff** that benefits the bondholder.
        """)

        ex_convex = compute_convexity(ex_yield, 10)
        st.markdown("**Worked Example: 10Y at {:.2f}%, +/-50bp shock**".format(ex_yield))

        dur_only_up = -ex_moddur / 100 * 0.50
        dur_only_dn = ex_moddur / 100 * 0.50
        full_up = dur_only_up + 0.5 * ex_convex * (0.005) ** 2 * 100
        full_dn = dur_only_dn + 0.5 * ex_convex * (0.005) ** 2 * 100

        st.code(f"""
Convexity = {ex_convex:.1f}

+50bp shock (Duration only):  {dur_only_up:.3f}%
+50bp shock (with Convexity): {full_up:.3f}%
  -> Convexity reduces the loss by {abs(full_up - dur_only_up):.3f}%

-50bp shock (Duration only):  +{dur_only_dn:.3f}%
-50bp shock (with Convexity): +{full_dn:.3f}%
  -> Convexity increases the gain by {abs(full_dn - dur_only_dn):.3f}%

Asymmetry: Gain from -50bp (+{full_dn:.3f}%) > Loss from +50bp ({full_up:.3f}%)
""", language="text")

        # --- Curve Spreads ---
        st.markdown("---")
        st.markdown("### 5. Yield Curve Spreads")
        st.markdown("""
| Spread | Formula | Signal |
|--------|---------|--------|
| **2s10s** | 10Y yield - 2Y yield | Classic recession/expansion indicator. Negative = inverted = recession signal |
| **5s30s** | 30Y yield - 5Y yield | Long-end steepness. Driven by term premium and supply |
| **3m10y** | 10Y yield - 3M yield | Fed-sensitive spread. Used in NY Fed recession model |
| **2s5s10s Butterfly** | 2 x 5Y - 2Y - 10Y | Belly richness/cheapness. Used for relative value trades |

**Steepening regimes:**

- **Bull steepener:** Short rates fall faster than long rates. Happens when the Fed cuts --
  the front end rallies on lower policy rates while the long end is anchored by term premium
  and growth/inflation expectations. The current normalization from the 2022-2024 inversion is
  a textbook bull steepener.

- **Bear steepener:** Long rates rise faster than short rates. Driven by inflation fears,
  fiscal supply concerns, or term premium expansion. Both ends may rise but the long end
  moves more.

- **Bull flattener:** Long rates fall faster than short rates (risk-off / flight to quality).

- **Bear flattener:** Short rates rise faster than long rates (Fed hiking cycle -- this is
  what created the 2022-2024 inversion).
        """)

        # --- Credit Spreads ---
        st.markdown("---")
        st.markdown("### 6. ICE BofA Credit Spreads (OAS)")
        st.markdown("""
**Option-Adjusted Spread (OAS)** is the yield spread of a corporate bond index
over a spot Treasury curve, adjusted for any embedded options (e.g. call provisions).

| Index | FRED Series | Rating | Description |
|-------|-------------|--------|-------------|
| IG OAS | BAMLC0A0CM | BBB or better | Investment grade corporates |
| BBB OAS | BAMLC0A4CBBB | BBB | Lowest investment grade |
| HY OAS | BAMLH0A0HYM2 | Below BB | High yield / junk bonds |
| AAA OAS | BAMLC0A1CAAA | AAA | Highest quality corporates |

**Key ratios:**
- **HY/IG Compression Ratio** = HY OAS / IG OAS. Lower = compressed = risk-on sentiment
- **BBB-IG Differential** = BBB OAS - IG OAS. Compensation for downgrade risk
        """)

        st.markdown("---")
        st.markdown("### 7. Data Sources")
        st.markdown("""
| Data | Source | FRED Series |
|------|--------|-------------|
| Treasury Yields | US Treasury Dept, H.15 Release | DGS1MO, DGS3MO, DGS6MO, DGS1, DGS2, DGS3, DGS5, DGS7, DGS10, DGS20, DGS30 |
| Credit Spreads | ICE Data Indices, LLC | BAMLC0A0CM, BAMLH0A0HYM2, BAMLC0A4CBBB, etc. |
| Fed Funds Rate | Federal Reserve | DFF |

All data is fetched live from the **FRED API** (Federal Reserve Bank of St. Louis).
No sample or synthetic data is used.
        """)


if __name__ == "__main__":
    main()
