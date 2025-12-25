import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.constants import BANK_TICKERS
from utils.data_loader import get_price_data, get_returns
from models.var_vecm import fit_var, johansen_test, fit_vecm

# ==================================================
# =============== ACCESS CONTROL ===================
# ==================================================
if st.session_state.get("mode") != "advanced":
    st.warning("This section is available under Advanced Analytics.")
    # st.stop()

if "banks" not in st.session_state or not st.session_state.banks:
    st.warning("Please select the organisations from the Home page.")
    st.stop()

selected_banks = st.session_state.banks

if len(selected_banks) < 2:
    st.warning("VAR / VECM requires at least two organisations.")
    st.stop()

tickers = [BANK_TICKERS[b] for b in selected_banks]

# ==================================================
# ================= PAGE SETUP =====================
# ==================================================
st.set_page_config(layout="wide")
st.title("🔗 VAR / VECM – Inter-Bank Dynamics")

# ==================================================
# ================= SIDEBAR ========================
# ==================================================
st.sidebar.image("assets/NMIMS_B.png", use_container_width=True)

st.sidebar.header("Model Configuration")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)
end_date = st.sidebar.date_input("End Date")

st.sidebar.subheader("Lag Configuration")

max_lags = st.sidebar.slider(
    "Maximum Lag Order",
    min_value=1,
    max_value=10,
    value=5
)

lag_criterion = st.sidebar.selectbox(
    "Lag Selection Criterion",
    ["aic", "bic", "hqic"]
)

st.sidebar.subheader("IRF Configuration")

irf_horizon = st.sidebar.slider(
    "IRF Horizon (Periods)",
    min_value=5,
    max_value=20,
    value=10
)

model_type = st.sidebar.radio(
    "Model Type",
    ["VAR", "VECM"]
)

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
# ======================= VAR ======================
# ==================================================
if model_type == "VAR":

    st.subheader("Vector Autoregression (VAR)")

    var_results = fit_var(
        returns,
        maxlags=max_lags,
        ic=lag_criterion
    )

    col1, col2 = st.columns(2)
    col1.metric("Selected Lag Order", var_results.k_ar)
    col2.metric(
        f"{lag_criterion.upper()}",
        round(getattr(var_results, lag_criterion), 2)
    )

    # ---------------- Stability ----------------
    st.markdown("### 🔍 Model Stability")

    if var_results.is_stable(verbose=False):
        st.success("VAR model is stable (all eigenvalues lie within the unit circle).")
    else:
        st.warning("VAR model may be unstable. Interpret results cautiously.")

    # ---------------- Summary ----------------
    with st.expander("📄 VAR Model Summary"):
        st.text(var_results.summary())

    # ---------------- IRF ----------------
    st.subheader("Impulse Response Functions (IRF)")

    irf = var_results.irf(irf_horizon)
    fig = irf.plot(orth=False)
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**  
    Impulse Response Functions illustrate how shocks to one bank’s returns
    propagate through the banking system over time, capturing
    short-run interdependencies.
    """)

# ==================================================
# ======================= VECM =====================
# ==================================================
if model_type == "VECM":

    st.subheader("Vector Error Correction Model (VECM)")

    # ---------------- Johansen Test ----------------
    johansen_res = johansen_test(
        returns,
        det_order=0,
        k_ar_diff=max_lags
    )

    trace_stats = johansen_res.lr1
    crit_90 = johansen_res.cvt[:, 0]
    crit_95 = johansen_res.cvt[:, 1]
    crit_99 = johansen_res.cvt[:, 2]

    johansen_df = pd.DataFrame({
        "Hypothesis (r ≤)": range(len(trace_stats)),
        "Trace Statistic": trace_stats.round(2),
        "90% Critical Value": crit_90.round(2),
        "95% Critical Value": crit_95.round(2),
        "99% Critical Value": crit_99.round(2)
    })

    st.markdown("### 🔍 Johansen Cointegration Test")
    st.dataframe(johansen_df, use_container_width=True)

    coint_rank = sum(trace_stats > crit_95)

    st.success(
        f"Estimated Number of Cointegrating Relations (5% level): **{coint_rank}**"
    )

    if coint_rank == 0:
        st.warning(
            "No cointegration detected at the 5% level. "
            "VECM may not be appropriate."
        )
        st.stop()

    # ---------------- VECM Fit ----------------
    vecm_results = fit_vecm(
        returns,
        coint_rank=coint_rank,
        k_ar_diff=max_lags
    )

    # ---------------- Long-Run ----------------
    st.subheader("🔗 Long-Run Equilibrium (Cointegrating Vectors)")

    beta = pd.DataFrame(
        vecm_results.beta,
        index=returns.columns,
        columns=[f"Cointegrating Vector {i+1}" for i in range(coint_rank)]
    )

    st.dataframe(beta.round(4), use_container_width=True)

    st.markdown("""
    **Long-run interpretation:**  
    Cointegrating vectors define the equilibrium relationship
    binding the selected organisations over time.
    """)

    # ---------------- Short-Run ----------------
    st.subheader("⚖️ Short-Run Adjustment (Error Correction Terms)")

    alpha = pd.DataFrame(
        vecm_results.alpha,
        index=returns.columns,
        columns=[f"ECM {i+1}" for i in range(coint_rank)]
    )

    st.dataframe(alpha.round(4), use_container_width=True)

    st.markdown("""
    **Short-run interpretation:**  
    - Significant negative coefficients indicate adjustment toward equilibrium  
    - Organisations with weak coefficients act as long-run leaders  
    - Others absorb short-run shocks
    """)

    # ---------------- Summary ----------------
    with st.expander("📄 Full VECM Model Summary"):
        st.text(vecm_results.summary())
