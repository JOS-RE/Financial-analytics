import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import get_price_data, get_returns
from models.garch import fit_garch

st.set_page_config(layout="wide")
st.title("🌪️ GARCH Volatility Modelling")

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
    "Select Banks",
    options=list(BANK_TICKERS.keys()),
    default=[
        "HDFC Bank",
        "ICICI Bank",
    ]
)

if len(selected_banks) == 0:
    st.warning("Please select at least one bank.")
    st.stop()

if len(selected_banks) > 7:
    st.warning("Please select up to 7 banks for optimal performance.")
    st.stop()

tickers = [BANK_TICKERS[bank] for bank in selected_banks]

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date")

# ---------------- Data ----------------
prices = get_price_data(tickers, start_date, end_date)
returns = get_returns(prices)

st.subheader("Daily Returns")

# Ensure proper column names
if returns.shape[1] == 1:
    returns.columns = [selected_banks[0]]

st.line_chart(returns)


# ---------------- GARCH ----------------
st.subheader("GARCH(1,1) Model Results by Bank")

bank_tabs = st.tabs(list(returns.columns))

for tab, bank in zip(bank_tabs, returns.columns):
    with tab:
        st.markdown(f"## 🏦 {bank}")

        bank_returns = returns[bank].dropna()
        garch_results = fit_garch(bank_returns)

        # ---- Key Parameters Table ----
        params = garch_results.params
        pvalues = garch_results.pvalues

        summary_df = pd.DataFrame({
            "Parameter": params.index,
            "Estimate": params.values.round(4),
            "P-Value": pvalues.values.round(4)
        })

        st.markdown("### 📌 Estimated Parameters")
        st.dataframe(summary_df, use_container_width=True)

        # ---- Model Diagnostics ----
        st.markdown("### 📊 Model Diagnostics")

        col1, col2, col3 = st.columns(3)
        col1.metric("Log Likelihood", round(garch_results.loglikelihood, 2))
        col2.metric("AIC", round(garch_results.aic, 2))
        col3.metric("BIC", round(garch_results.bic, 2))

        # ---- Full Summary (Expandable) ----
        with st.expander(f"📄 Full GARCH Summary – {bank}"):
            st.text(garch_results.summary())

        # ---- Volatility Plot ----
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(garch_results.conditional_volatility, label="Conditional Volatility")
        ax.set_title(f"Estimated Conditional Volatility – {bank}")
        ax.set_ylabel("Volatility")
        ax.set_xlabel("Time")
        ax.legend()

        st.pyplot(fig)

# ---------------- Interpretation ----------------
st.markdown("""
### 📌 Interpretation
- Separate GARCH(1,1) models are estimated for each bank
- Volatility exhibits **clustering and persistence**
- Differences across banks reflect **heterogeneous risk dynamics**
""")
