import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy.stats import f

from utils.constants import BANK_TICKERS
from utils.data_loader import get_price_data, get_returns
from models.var_vecm import fit_var

# ==================================================
# =============== ACCESS CONTROL ===================
# ==================================================
if st.session_state.get("mode") != "advanced":
    st.warning("Please start from the Home page.")
    # st.stop()

if "banks" not in st.session_state or len(st.session_state.banks) < 2:
    st.warning("Please select at least two banks from the Home page.")
    st.stop()

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("⏱️ Distributed Lag Model (DLM)")

st.caption(
    "Objective: Analyse how lagged returns of multiple banks affect the "
    "current returns of a target bank, and compare DLM with VAR dynamics."
)

# ==================================================
# ================= SIDEBAR ========================
# ==================================================
st.sidebar.header("DLM Inputs")

selected_banks = st.session_state.banks

dep_bank = st.sidebar.selectbox(
    "Dependent Bank (Y)",
    selected_banks
)

indep_banks = st.sidebar.multiselect(
    "Independent Banks (X)",
    [b for b in selected_banks if b != dep_bank],
    default=[b for b in selected_banks if b != dep_bank][:1]
)

if not indep_banks:
    st.warning("Please select at least one independent bank.")
    st.stop()

max_lags = st.sidebar.slider(
    "Number of Lags (k)",
    min_value=1,
    max_value=10,
    value=4
)

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("End Date")

# ==================================================
# ================= DATA ===========================
# ==================================================
tickers = [BANK_TICKERS[b] for b in [dep_bank] + indep_banks]

prices = get_price_data(tickers, start_date, end_date)

if prices is None or prices.empty:
    st.error("No data available.")
    st.stop()

returns = get_returns(prices)
returns.columns = [dep_bank] + indep_banks
returns = returns.dropna()

y = returns[dep_bank]

# ==================================================
# ============ UNRESTRICTED DLM ====================
# ==================================================
st.subheader("📌 Unrestricted Distributed Lag Model (Spillovers)")

X_lagged = []

for bank in indep_banks:
    for lag in range(1, max_lags + 1):
        X_lagged.append(returns[bank].shift(lag))

X_unres = pd.concat(X_lagged, axis=1)
X_unres.columns = [
    f"{bank}_Lag{lag}"
    for bank in indep_banks
    for lag in range(1, max_lags + 1)
]

df_unres = pd.concat([y, X_unres], axis=1).dropna()

Y_u = df_unres.iloc[:, 0]
X_u = sm.add_constant(df_unres.iloc[:, 1:])

unrestricted_model = sm.OLS(Y_u, X_u).fit()

with st.expander("📄 Unrestricted Model Summary"):
    st.text(unrestricted_model.summary())


# ==================================================
# ============== RESTRICTED DLM ====================
# ==================================================
st.subheader("📌 Restricted Distributed Lag Model (Own Dynamics)")

Y_lags = pd.concat(
    [y.shift(i) for i in range(1, max_lags + 1)],
    axis=1
)
Y_lags.columns = [f"Y_Lag{i}" for i in range(1, max_lags + 1)]

df_res = pd.concat([y, Y_lags], axis=1).dropna()

Y_r = df_res.iloc[:, 0]
X_r = sm.add_constant(df_res.iloc[:, 1:])

restricted_model = sm.OLS(Y_r, X_r).fit()

with st.expander("📄 Restricted Model Summary"):
    st.text(restricted_model.summary())


# ==================================================
# ================== F TEST ========================
# ==================================================
st.subheader("📊 F-Test: Restricted vs Unrestricted")

RSS_r = restricted_model.ssr
RSS_u = unrestricted_model.ssr

n = unrestricted_model.nobs
k = len(indep_banks) * max_lags

F_stat = ((RSS_r - RSS_u) / (k - len(indep_banks))) / (RSS_u / (n - k - 1))
p_value = 1 - f.cdf(F_stat, k - len(indep_banks), n - k - 1)

col1, col2 = st.columns(2)
col1.metric("F Statistic", round(F_stat, 4))
col2.metric("p-value", round(p_value, 4))

# ==================================================
# ============ MEAN & WEIGHTED LAG =================
# ==================================================
st.subheader("⏳ Lag Structure Summary")

lags = np.arange(1, max_lags + 1)

# Extract unrestricted lag coefficients (exclude constant)
coef_series = unrestricted_model.params.drop("const")
coef_matrix = coef_series.values.reshape(len(indep_banks), max_lags)

for i, bank in enumerate(indep_banks):

    betas = coef_matrix[i]

    # Mean lag (depends only on lag length)
    mean_lag = lags.mean()

    # Weighted lag (economic timing)
    weighted_lag = (
        np.sum(lags * betas) / np.sum(betas)
        if np.sum(betas) != 0
        else np.nan
    )

    st.markdown(f"**Spillover from {bank} → {dep_bank}**")

    col1, col2 = st.columns(2)

    col1.metric(
        "Mean Lag",
        round(mean_lag, 2)
    )

    col2.metric(
        "Weighted Lag",
        round(weighted_lag, 2)
    )


# ==================================================
# ================== DLM vs VAR ====================
# ==================================================
st.subheader("🔍 DLM vs VAR Comparison")

var_results = fit_var(returns[[dep_bank] + indep_banks], maxlags=max_lags, ic="aic")

comparison_df = pd.DataFrame({
    "Model": ["DLM (Unrestricted)", "VAR"],
    "AIC": [unrestricted_model.aic, var_results.aic],
    "BIC": [unrestricted_model.bic, var_results.bic],
    "Structure": ["Directional", "System-wide"]
})

st.dataframe(comparison_df, use_container_width=True)

# ==================================================
# ================= INTERPRETATION =================
# ==================================================
st.markdown("""
### 📌 Interpretation

- The **unrestricted DLM** allows each bank’s lagged returns to have
  a distinct and time-specific impact on the dependent bank.
- The **restricted DLM** enforces a smoother lag structure, testing
  whether timing granularity is statistically necessary.
- The **F-test** evaluates whether the richer lag specification
  significantly improves explanatory power.
- **Weighted lag estimates** identify when spillover effects are strongest.
- Compared to VAR:
  - **DLM** provides clearer lag interpretation and directionality.
  - **VAR** captures feedback effects but with reduced lag clarity.
- Together, they offer a **comprehensive view of inter-bank transmission**.
""")
