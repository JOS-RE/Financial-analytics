import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import BANK_TICKERS
from utils.data_loader import get_price_data, get_returns
from models.algo_trading import (
    sma_long_only, sma_short_only, sma_long_short,
    rsi_long_only, rsi_short_only, rsi_long_short,
    custom_triple_sma
)

# ==================================================
# =============== ACCESS CONTROL ===================
# ==================================================
if st.session_state.get("mode") != "trading":
    st.warning("You did not select trading mode from the Home page.")
    # st.stop()

if "banks" not in st.session_state or not st.session_state.banks:
    st.warning("Please select banks from the Home page.")
    st.stop()

st.caption("Workflow: Volatility → Trading Signals")

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("🤖 Algorithmic Trading Strategies")

# ==================================================
# =============== BANK SELECTION ===================
# ==================================================
selected_banks = st.session_state.banks

st.sidebar.subheader("Bank Selection")

bank = st.sidebar.selectbox(
    "Select Bank for Trading",
    options=selected_banks
)

ticker = BANK_TICKERS[bank]

# ==================================================
# ================= SIDEBAR ========================
# ==================================================
st.sidebar.subheader("Strategy Inputs")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2021-01-01")
)
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
# ================= DATA ===========================
# ==================================================
prices = get_price_data(ticker, start_date, end_date)

if prices is None or prices.empty:
    st.error("No price data available for the selected inputs.")
    st.stop()

prices.index = pd.to_datetime(prices.index)
price_series = prices.iloc[:, 0].astype(float).dropna()

returns = get_returns(price_series.to_frame()).iloc[:, 0].dropna()

# ---------------- Buy & Hold Benchmark ----------------
bh_ann_return = ((returns.mean() + 1) ** 252 - 1) * 100
bh_std = returns.std() * (252 ** 0.5) * 100

# ==================================================
# ================= PRICE PLOT =====================
# ==================================================
st.subheader(f"📈 Price Series – {bank}")

fig, ax = plt.subplots(figsize=(10, 4))

ax.plot(price_series, color="#F57C00", linewidth=2)
ax.fill_between(
    price_series.index,
    price_series,
    price_series.rolling(20).min(),
    color="#F57C00",
    alpha=0.15
)

ax.set_title(f"{bank} Price Movement")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.grid(alpha=0.2)

st.pyplot(fig)

# ==================================================
# =============== STRATEGY EXECUTION ===============
# ==================================================
results = []

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
# =============== SUMMARY TABLE ====================
# ==================================================
st.subheader("📊 Strategy Performance Summary")

summary_df = pd.DataFrame([
    {
        "Strategy": r["Strategy"],
        "Annualized Return (%)": f"{r['AnnualizedReturn']:.2f}%",
        "Volatility (Std Dev %)": f"{r['StdDev']:.2f}%",
        "Sharpe Ratio": round(r["SharpeRatio"], 3),
        "Number of Trades": int(r["Trades"])
    }
    for r in results
])

st.dataframe(summary_df, use_container_width=True)

# ==================================================
# =============== COMPARISON PLOTS =================
# ==================================================
if len(results) > 1:
    st.subheader("📈 Strategy Comparison")

    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            summary_df["Strategy"],
            [float(x.replace("%", "")) for x in summary_df["Annualized Return (%)"]],
            color="#FB8C00",
            alpha=0.85
        )
        ax.set_title("Annualized Return (%)")
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
        ax.set_title("Sharpe Ratio")
        st.pyplot(fig)

# ==================================================
# =============== RISK–RETURN ======================
# ==================================================
if len(results) > 1:
    st.subheader("⚖️ Risk–Return Comparison")

    fig, ax = plt.subplots(figsize=(4, 3))

    ax.scatter(
        [r["StdDev"] for r in results],
        [r["AnnualizedReturn"] for r in results],
        s=100,
        label="Strategies"
    )

    ax.scatter(
        bh_std,
        bh_ann_return,
        marker="X",
        s=120,
        label="Buy & Hold"
    )

    for r in results:
        ax.annotate(
            r["Strategy"],
            (r["StdDev"], r["AnnualizedReturn"]),
            fontsize=7,
            xytext=(5, 5),
            textcoords="offset points"
        )

    ax.set_xlabel("Risk (Std Dev %)")
    ax.set_ylabel("Annualized Return (%)")
    ax.legend()

    st.pyplot(fig)

# ==================================================
# ============ STRATEGY DETAILS ====================
# ==================================================
st.subheader("🔍 Strategy Details")
st.info(
    "⬇️ Expand the sections below to explore **strategy-level signals, indicators, "
    "and cumulative performance vs Buy & Hold**."
)


for r in results:
    with st.expander(f"📌 {r['Strategy']}"):
        df = r["Data"].copy()

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["Price"], label="Price", color="black")
            for col in df.columns:
                if col.startswith("SMA"):
                    ax.plot(df[col], linestyle="--", label=col)
            ax.legend()
            ax.set_title("Price & Indicators")
            st.pyplot(fig)

        with col2:
            if "RSI" in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df["RSI"], color="purple")
                ax.axhline(70, linestyle="--", color="red")
                ax.axhline(30, linestyle="--", color="green")
                ax.set_title("RSI Indicator")
                st.pyplot(fig)
            else:
                df["Strategy Line"] = (1 + df["AlgoRet"]).cumprod() - 1
                df["Buy & Hold Line"] = (1 + df["Returns"]).cumprod() - 1

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(df["Strategy Line"], label="Strategy")
                ax.plot(df["Buy & Hold Line"], label="Buy & Hold")
                ax.legend()
                ax.set_title("Cumulative Returns")
                st.pyplot(fig)

# ==================================================
# =============== INTERPRETATION ===================
# ==================================================
st.markdown("""
### 📌 Interpretation

**Strategy Line**
- Represents cumulative returns from rule-based trading.
- Capital is deployed only when signals are active.
- Captures timing and execution efficiency.

**Buy & Hold Line**
- Represents passive investment over the same period.
- Serves as the benchmark for performance comparison.

**Key Insight**
- Strategy outperforming Buy & Hold indicates alpha.
- Smoother curves imply risk reduction even with similar returns.
""")
