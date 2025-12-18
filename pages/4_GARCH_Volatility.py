import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd

from utils.data_loader import get_price_data, get_returns
from models.garch import fit_garch

st.set_page_config(layout="wide")
st.title("🌪️ GARCH Volatility Modelling")

# ---------------- Sidebar ----------------
st.sidebar.header("Inputs")

ticker = st.sidebar.selectbox(
    "Select Bank",
    ["HDFCBANK.NS", "ICICIBANK.NS", "SBIN.NS"]
)

start_date = st.sidebar.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
end_date = st.sidebar.date_input("End Date")

# ---------------- Data ----------------
prices = get_price_data(ticker, start_date, end_date)
returns = get_returns(prices)

st.subheader("Daily Returns")
returns.columns = ["Daily Returns"]
st.line_chart(returns)




# ---------------- GARCH ----------------
# st.subheader("GARCH(1,1) Results")
# garch_results = fit_garch(returns)
# st.text(garch_results.summary())

st.subheader("GARCH(1,1) Model Results")

garch_results = fit_garch(returns)

# ---- Key Parameters Table ----
params = garch_results.params
pvalues = garch_results.pvalues

summary_df = (
    pd.DataFrame({
        "Parameter": params.index,
        "Estimate": params.values.round(4),
        "P-Value": pvalues.values.round(4)
    })
)

st.markdown("### 📌 Estimated Parameters")
st.dataframe(summary_df, use_container_width=True)

# ---- Model Diagnostics ----
st.markdown("### 📊 Model Diagnostics")

col1, col2, col3 = st.columns(3)

col1.metric("Log Likelihood", round(garch_results.loglikelihood, 2))
col2.metric("AIC", round(garch_results.aic, 2))
col3.metric("BIC", round(garch_results.bic, 2))

# ---- Full Summary (Expandable) ----
with st.expander("📄 Full GARCH Summary"):
    st.text(garch_results.summary())


# ---------------- Volatility Plot ----------------
st.subheader("Conditional Volatility")

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(garch_results.conditional_volatility)
ax.set_title("Estimated Conditional Volatility")
ax.set_ylabel("Volatility")
ax.set_xlabel("Time")

st.pyplot(fig)

# ---------------- Interpretation ----------------
st.markdown("""
### 📌 Interpretation
- Volatility is **time-varying** and shows **clustering**
- Periods of high volatility are followed by high volatility
- GARCH effectively captures **risk persistence** in bank returns
""")
