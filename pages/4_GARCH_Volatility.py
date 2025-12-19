import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils.constants import BANK_TICKERS
from utils.data_loader import get_price_data, get_returns
from models.garch import fit_garch

# ==================================================
# =============== ACCESS CONTROL ===================
# ==================================================
if st.session_state.get("mode") != "trading":
    st.warning("You did not select trading mode from the Home page.")
    # st.stop()

if "banks" not in st.session_state or not st.session_state.banks:
    st.warning("Please select banks from the Home page.")
    st.stop()

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("🌪️ GARCH Volatility Modelling")
st.caption("Workflow: Volatility → Trading Signals")

# ==================================================
# ================= SIDEBAR ========================
# ==================================================
st.sidebar.header("Model Inputs")

# ---- Bank selector (override Home selection) ----
selected_banks_sidebar = st.sidebar.multiselect(
    "Select Banks",
    options=list(BANK_TICKERS.keys()),
    default=st.session_state.banks
)

if not selected_banks_sidebar:
    st.sidebar.warning("Please select at least one bank.")
    st.stop()

# Update session state so next page inherits selection
st.session_state.banks = selected_banks_sidebar
selected_banks = selected_banks_sidebar

# ---- Date range ----
start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("End Date")

# ==================================================
# ================= DATA ===========================
# ==================================================
tickers = [BANK_TICKERS[b] for b in selected_banks]

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
# ================= GARCH MODELS ===================
# ==================================================
st.subheader("GARCH(1,1) Results by Bank")

bank_tabs = st.tabs(list(returns.columns))

for tab, bank in zip(bank_tabs, returns.columns):
    with tab:
        st.markdown(f"## 🏦 {bank}")

        bank_returns = returns[bank].dropna()

        if bank_returns.empty:
            st.warning("Insufficient data for GARCH estimation.")
            continue

        garch_results = fit_garch(bank_returns)

        # ---------------- Parameters ----------------
        params = garch_results.params
        pvalues = garch_results.pvalues

        summary_df = pd.DataFrame({
            "Parameter": params.index,
            "Estimate": params.values.round(4),
            "P-Value": pvalues.values.round(4)
        })

        st.markdown("### 📌 Estimated Parameters")
        st.dataframe(summary_df, use_container_width=True)

        # ---------------- Diagnostics ----------------
        st.markdown("### 📊 Model Diagnostics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Log Likelihood", round(garch_results.loglikelihood, 2))
        col2.metric("AIC", round(garch_results.aic, 2))
        col3.metric("BIC", round(garch_results.bic, 2))

        # ---------------- Full Summary ----------------
        with st.expander(f"📄 Full GARCH Summary – {bank}"):
            st.text(garch_results.summary())

        # ---------------- Volatility Plot ----------------
        fig, ax = plt.subplots(figsize=(10, 4))

        ax.plot(
            garch_results.conditional_volatility,
            color="#F57C00",
            linewidth=1.8,
            label="Conditional Volatility"
        )

        ax.fill_between(
            garch_results.conditional_volatility.index,
            garch_results.conditional_volatility.values,
            color="#F57C00",
            alpha=0.2
        )

        ax.set_title(f"Estimated Conditional Volatility – {bank}")
        ax.set_ylabel("Volatility")
        ax.set_xlabel("Time")
        ax.grid(alpha=0.3)
        ax.legend()

        st.pyplot(fig)

# ==================================================
# ================= INTERPRETATION =================
# ==================================================
st.markdown("""
### 📌 Interpretation
- Separate **GARCH(1,1)** models are estimated for each bank.
- Volatility shows **clustering and persistence**, a core stylised fact of returns.
- Differences across banks reflect **heterogeneous risk profiles**.
- These volatility estimates can directly inform **risk-aware trading strategies**.
""")

# ==================================================
# ================= NAVIGATION =====================
# ==================================================
st.divider()

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("➡️ Continue to Algorithmic Trading"):
        st.switch_page("pages/1_Algorithmic_Trading.py")
