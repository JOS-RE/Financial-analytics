import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from models.portfolio import random_portfolios


from utils.data_loader import get_price_data, get_returns
from models.portfolio import (
    min_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier
)

st.set_page_config(layout="wide")
st.title("📈 Portfolio Optimisation (Long-Only)")

# ---------------- Sidebar ----------------
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

selected_banks = st.sidebar.multiselect(
    "Select Banks (Min 2)",
    options=list(BANK_TICKERS.keys()),
    default=[
        "HDFC Bank",
        "ICICI Bank",
        "State Bank of India"
    ]
)

if len(selected_banks) < 2:
    st.warning("Please select at least two banks.")
    st.stop()

tickers = [BANK_TICKERS[b] for b in selected_banks]

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("End Date")

rf = st.sidebar.number_input(
    "Risk-Free Rate (Annual)",
    min_value=0.0,
    max_value=0.15,
    value=0.0625,
    step=0.005
)

rf_daily = rf / 252

# ---------------- Data ----------------
prices = get_price_data(tickers, start_date, end_date)
returns = get_returns(prices)
returns.columns = selected_banks

st.subheader("Daily Returns")
st.line_chart(returns)

# ==================================================
# ================= OPTIMISATION ===================
# ==================================================
st.subheader("Optimised Portfolios")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔹 Minimum Variance Portfolio")
    w_minvar = min_variance_portfolio(returns)
    st.dataframe(w_minvar.round(4), use_container_width=True)

with col2:
    st.markdown("### 🔹 Maximum Sharpe Ratio Portfolio")
    w_sharpe = max_sharpe_portfolio(returns, rf_daily)
    st.dataframe(w_sharpe.round(4), use_container_width=True)

# ==================================================
# ============ WEIGHT DISTRIBUTION =================
# ==================================================
st.subheader("📊 Portfolio Weight Distribution")

def plot_weights(weights, title):
    fig, ax = plt.subplots(figsize=(8, 4))
    weights.plot(kind="bar", ax=ax)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_xlabel("Asset")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)

col1, col2 = st.columns(2)

with col1:
    plot_weights(w_minvar, "Minimum Variance Portfolio Weights")

with col2:
    plot_weights(w_sharpe, "Maximum Sharpe Portfolio Weights")

# ==================================================
# ============== PORTFOLIO CONCENTRATION ===========
# ==================================================
st.subheader("📐 Portfolio Concentration")

def portfolio_concentration(weights):
    return (weights ** 2).sum()

hhi_minvar = portfolio_concentration(w_minvar)
hhi_sharpe = portfolio_concentration(w_sharpe)

col1, col2 = st.columns(2)

col1.metric(
    "HHI – Min Variance Portfolio",
    round(hhi_minvar, 4)
)

col2.metric(
    "HHI – Max Sharpe Portfolio",
    round(hhi_sharpe, 4)
)

st.markdown("""
**Interpretation:**  
- The Herfindahl–Hirschman Index (HHI) measures portfolio concentration.  
- Higher values indicate **greater concentration** in fewer assets.  
- Lower values indicate **better diversification**.  
- Comparing HHI across portfolios highlights the diversification trade-off
  between risk minimisation and return maximisation.
""")

# ==================================================
# ===== RISK–RETURN–CONCENTRATION SCATTER ==========
# ==================================================
st.subheader("🟣 Risk–Return–Concentration Landscape")

st.markdown("""
Each point represents a feasible long-only portfolio.
Colour intensity reflects **portfolio concentration (HHI)**.
""")

rand_ports = random_portfolios(returns, n_portfolios=3000)

fig, ax = plt.subplots(figsize=(9, 6))

scatter = ax.scatter(
    rand_ports["Risk"],
    rand_ports["Return"],
    c=rand_ports["Concentration"],
    cmap="YlOrBr",
    alpha=0.6
)

ax.set_xlabel("Portfolio Risk (Std Dev)")
ax.set_ylabel("Portfolio Return")
ax.set_title("Portfolio Risk–Return Space Coloured by Concentration")

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label("Concentration (HHI)")

st.pyplot(fig)


# ==================================================
# ================ EFFICIENT FRONTIER ==============
# ==================================================
st.subheader("Efficient Frontier")

ef = efficient_frontier(returns)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ef["Risk"], ef["Return"], marker="o", linestyle="-")
ax.set_xlabel("Portfolio Risk (Std Dev)")
ax.set_ylabel("Portfolio Return")
ax.set_title("Efficient Frontier (Long-Only)")

st.pyplot(fig)



# ---------------- Interpretation ----------------
st.markdown("""
### 📌 Overall Interpretation
- Portfolios are constructed under **full-investment** and **long-only** constraints.
- The **minimum variance portfolio** focuses on risk reduction through diversification.
- The **maximum Sharpe portfolio** optimises risk-adjusted returns using a
  configurable risk-free rate.
- Weight distribution and concentration metrics provide insight into
  **portfolio composition and diversification quality**.
- The efficient frontier illustrates the achievable **risk–return trade-off**
  across optimal portfolios.
""")
