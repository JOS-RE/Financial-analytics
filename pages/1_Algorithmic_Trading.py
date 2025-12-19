import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import get_price_data, get_returns
from models.algo_trading import (
    sma_long_only, sma_short_only, sma_long_short,
    rsi_long_only, rsi_short_only, rsi_long_short,
    custom_triple_sma
)

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("🤖 Algorithmic Trading Strategies")

# ==================================================
# ================= SIDEBAR =========================
# ==================================================
st.sidebar.header("Inputs")

BANK_TICKERS = {
    "HDFC Bank": "HDFCBANK.NS",
    "ICICI Bank": "ICICIBANK.NS",
    "State Bank of India": "SBIN.NS",
    "Axis Bank": "AXISBANK.NS",
    "Kotak Mahindra Bank": "KOTAKBANK.NS",
    "IndusInd Bank": "INDUSINDBK.NS",
    "Bank of Baroda": "BANKBARODA.NS"
}

bank = st.sidebar.selectbox(
    "Select Bank",
    list(BANK_TICKERS.keys()),
    index=2
)

ticker = BANK_TICKERS[bank]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
end_date = st.sidebar.date_input("End Date")

rf = st.sidebar.number_input(
    "Risk-Free Rate (Annual)",
    min_value=0.0,
    max_value=0.15,
    value=0.065,
    step=0.005
)

strategy_group = st.sidebar.selectbox(
    "Strategy Group",
    ["SMA (7,14)", "RSI (14, 30–70)", "Custom Triple SMA"]
)

# ==================================================
# ================= DATA ============================
# ==================================================

prices = get_price_data(ticker, start_date, end_date)

if prices is None or prices.empty:
    st.error("No price data available for the selected inputs.")
    st.stop()

# Ensure datetime index
prices.index = pd.to_datetime(prices.index)

# Extract clean price series
price_series = prices.iloc[:, 0].astype(float).dropna()

# Compute returns
returns = get_returns(price_series.to_frame()).iloc[:, 0].dropna()

# ---------------- Buy & Hold Benchmark ----------------
bh_ann_return = ((returns.mean() + 1) ** 252 - 1) * 100
bh_std = returns.std() * (252 ** 0.5) * 100
# ---------------- Price Plot (Minimalist) ----------------
st.subheader(f"📈 Price Series – {bank}")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(
    price_series.index,
    price_series.values,
    color="#F57C00",
    linewidth=2
)

ax.fill_between(
    price_series.index,
    price_series.values,
    price_series.rolling(20).min(),
    color="#F57C00",
    alpha=0.15
)

ax.set_title(f"{bank} Price Movement")
ax.set_xlabel("")
ax.set_ylabel("Price")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)

st.pyplot(fig)



# ==================================================
# =============== STRATEGY EXECUTION =================
# ==================================================
results = []

price_series = prices.iloc[:, 0]

if strategy_group == "SMA (7,14)":
    results.append(sma_long_only(price_series, returns, rf=rf))
    results.append(sma_short_only(price_series, returns, rf=rf))
    results.append(sma_long_short(price_series, returns, rf=rf))

elif strategy_group == "RSI (14, 30–70)":
    results.append(rsi_long_only(price_series, returns, rf=rf))
    results.append(rsi_short_only(price_series, returns, rf=rf))
    results.append(rsi_long_short(price_series, returns, rf=rf))

else:
    results.append(custom_triple_sma(price_series, returns, rf=rf))

# ==================================================
# =============== SUMMARY TABLE =====================
# ==================================================
st.subheader("📊 Strategy Performance Summary")

summary_df = pd.DataFrame([
    {
        "Strategy": r["Strategy"],
        "Annualized Return (%)": round(r["AnnualizedReturn"], 2),
        "Std Dev (%)": round(r["StdDev"], 2),
        "Sharpe Ratio": round(r["SharpeRatio"], 3),
        "Number of Trades": int(r["Trades"])
    }
    for r in results
])

st.dataframe(summary_df, use_container_width=True)

# ==================================================
# =============== KEY METRIC (SINGLE STRATEGY) ======
# ==================================================
if len(summary_df) == 1:
    st.subheader("📌 Key Performance Metric")
    st.metric(
        "Sharpe Ratio",
        summary_df["Sharpe Ratio"].iloc[0]
    )

# ==================================================
# =============== COMPARISON PLOTS ==================
# ==================================================
if len(summary_df) > 1:
    st.subheader("📈 Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            summary_df["Strategy"],
            summary_df["Annualized Return (%)"],
            color="#FB8C00",
            alpha=0.85
        )
        ax.set_title("Annualized Return (%)", fontweight="bold")

        st.pyplot(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            summary_df["Strategy"],
            summary_df["Sharpe Ratio"],
            color="#FB8C00",
            alpha=0.85
        )
        ax.axvline(0, linestyle="--", color="gray", alpha=0.6)
        ax.set_title("Sharpe Ratio", fontweight="bold")

        st.pyplot(fig)

# ==================================================
# =============== RISK–RETURN SCATTER ===============
# ==================================================
if len(summary_df) > 1:
    st.subheader("⚖️ Risk–Return Comparison")

    fig, ax = plt.subplots(figsize=(3, 2))

    # Strategy points
    ax.scatter(
        summary_df["Std Dev (%)"],
        summary_df["Annualized Return (%)"],
        s=100,
        label="Strategies"
    )

    # Buy & Hold
    ax.scatter(
        bh_std,
        bh_ann_return,
        marker="X",
        s=120,
        label="Buy & Hold"
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["Strategy"],
            (row["Std Dev (%)"], row["Annualized Return (%)"]),
            fontsize=6,
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel("Risk (Std Dev %)")
    ax.set_ylabel("Annualized Return (%)")
    ax.set_title("Risk–Return Profile")
    ax.legend()

    st.pyplot(fig)
else:
    st.info(
        "Risk–return plots are most meaningful in a comparative setting. "
        "For a single strategy, refer to Sharpe ratio and cumulative returns below."
    )

# ==================================================
# ============ STRATEGY DETAILS =====================
# ==================================================
st.subheader("🔍 Strategy Details")

for r in results:
    with st.expander(f"📌 {r['Strategy']}"):
        df = r["Data"].copy()

        col1, col2 = st.columns(2)

        # ---------- Price & Indicators ----------
        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["Price"], label="Price", color="black")

            for col in df.columns:
                if col.startswith("SMA"):
                    ax.plot(df[col], linestyle="--", label=col)

            ax.set_title("Price & Indicators")
            ax.legend()
            st.pyplot(fig)

        # ---------- RSI or Cumulative Returns ----------
        with col2:
            if "RSI" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df["RSI"], color="purple", label="RSI")
                ax.axhline(70, linestyle="--", color="red")
                ax.axhline(30, linestyle="--", color="green")
                ax.set_title("RSI Indicator")
                st.pyplot(fig)
            else:
                df["Cumulative Strategy Return"] = (1 + df["AlgoRet"]).cumprod() - 1
                df["Cumulative Buy & Hold"] = (1 + df["Returns"]).cumprod() - 1

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df["Cumulative Strategy Return"], label="Strategy")
                ax.plot(df["Cumulative Buy & Hold"], label="Buy & Hold")
                ax.legend()
                ax.set_title("Cumulative Returns")
                st.pyplot(fig)

# ==================================================
# =============== INTERPRETATION ====================
# ==================================================
st.markdown("""
### 📌 Interpretation
- Technical trading strategies are evaluated under consistent assumptions.
- Long-only, short-only, and long–short variants highlight directional sensitivity.
- Risk–return analysis is presented **only in comparative settings**.
- Single-strategy evaluation focuses on **Sharpe ratio and benchmark comparison**.
- Detailed plots reveal signal behaviour, holding periods, and robustness.
""")
