import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import BANK_TICKERS
from utils.data_loader import get_price_data, get_returns
from models.portfolio import (
    min_variance_portfolio,
    max_sharpe_portfolio,
    efficient_frontier,
    random_portfolios
)

# ==================================================
# =============== ACCESS CONTROL ===================
# ==================================================
# if st.session_state.get("mode") != "long_term":
#     st.warning("Please start from the Home page.")
#     st.stop()

if "banks" not in st.session_state or not st.session_state.banks:
    st.warning("Please select the organisations from the Home page.")
    st.stop()

selected_banks = st.session_state.banks

if len(selected_banks) < 2:
    st.warning("Please select at least two Orgs for portfolio optimisation.")
    st.stop()

tickers = [BANK_TICKERS[b] for b in selected_banks]

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("📈 Portfolio Optimisation (Long-Only)")

# ==================================================
# ================= SIDEBAR ========================
# ==================================================
st.sidebar.image("assets/NMIMS_B.png", use_container_width=True)

st.sidebar.header("Optimisation Inputs")

# -------- Bank Selection (Portfolio-only) --------
st.sidebar.subheader("Bank Selection")

portfolio_banks = st.sidebar.multiselect(
    "Select Banks for Portfolio",
    options=list(BANK_TICKERS.keys()),
    default=st.session_state.banks,
    key="portfolio_bank_selector"
)

if len(portfolio_banks) < 2:
    st.sidebar.warning("Select at least two Orgs.")
    st.stop()

# -------- Date Range --------
start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("End Date")

# -------- Risk-Free Rate --------
rf = st.sidebar.number_input(
    "Risk-Free Rate (Annual)",
    min_value=0.0,
    max_value=0.15,
    value=0.0625,
    step=0.005
)

rf_daily = rf / 252

selected_banks = portfolio_banks
tickers = [BANK_TICKERS[b] for b in selected_banks]

st.sidebar.image("assets/logo2.png", use_container_width=True)
st.sidebar.markdown("---")


# ==================================================
# ================= DATA ============================
# ==================================================
prices = get_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("No price data available for the selected inputs.")
    st.stop()

returns = get_returns(prices)
returns.columns = selected_banks

# ==================================================
# ================= RETURNS PLOT ===================
# ==================================================
st.subheader("📊 Daily Returns")

st.line_chart(returns)

# ==================================================
# ================= OPTIMISATION ===================
# ==================================================
st.subheader("Optimised Portfolios")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔹 Minimum Variance Portfolio")
    w_minvar = min_variance_portfolio(returns)
    st.dataframe(w_minvar.round(4) * 100, use_container_width=True)

with col2:
    st.markdown("### 🔹 Maximum Sharpe Ratio Portfolio")
    w_sharpe = max_sharpe_portfolio(returns, rf_daily)
    st.dataframe(w_sharpe.round(4) * 100, use_container_width=True)

# ==================================================
# ============ WEIGHT DISTRIBUTION =================
# ==================================================
st.subheader("📊 Portfolio Weight Distribution")

def plot_weights(weights, title):
    fig, ax = plt.subplots(figsize=(7, 4))
    weights.plot(kind="bar", ax=ax, color="#FB8C00", alpha=0.85)
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_ylim(0, 1)
    ax.grid(axis="y", alpha=0.3)
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

col1.metric("HHI – Min Variance Portfolio", round(hhi_minvar, 4))
col2.metric("HHI – Max Sharpe Portfolio", round(hhi_sharpe, 4))

st.markdown("""
**Interpretation:**  
- The Herfindahl–Hirschman Index (HHI) measures portfolio concentration.  
- Higher values indicate **greater concentration**.  
- Lower values imply **better diversification**.
""")

# ==================================================
# ===== RISK–RETURN–CONCENTRATION LANDSCAPE =========
# ==================================================
st.subheader("🟣 Risk–Return–Concentration Landscape")

st.markdown("""
Each point represents a feasible **long-only portfolio**.  
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
ax.plot(ef["Risk"], ef["Return"], marker="o", linestyle="-", color="#F57C00")
ax.set_xlabel("Portfolio Risk (Std Dev)")
ax.set_ylabel("Portfolio Return")
ax.set_title("Efficient Frontier (Long-Only)")
ax.grid(alpha=0.3)

st.pyplot(fig)

# ==================================================
# ================= INTERPRETATION =================
# ==================================================
st.markdown("""
### 📌 Overall Interpretation
- Portfolios are constructed under **full-investment** and **long-only** constraints.
- The **minimum variance portfolio** prioritises risk reduction.
- The **maximum Sharpe portfolio** optimises risk-adjusted returns.
- Concentration metrics quantify diversification quality.
- The efficient frontier highlights the achievable **risk–return trade-off**.
""")
