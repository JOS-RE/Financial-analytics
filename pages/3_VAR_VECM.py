import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from utils.data_loader import get_price_data, get_returns
from models.var_vecm import fit_var, johansen_test, fit_vecm

st.set_page_config(layout="wide")
st.title("🔗 VAR / VECM – Inter-Bank Dynamics")

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

# -------- Bank Selection --------
selected_banks = st.sidebar.multiselect(
    "Select Banks (Min 2)",
    options=list(BANK_TICKERS.keys()),
    default=[
        "HDFC Bank",
        "ICICI Bank"
    ]
)

if len(selected_banks) < 2:
    st.warning("Please select at least two banks for VAR / VECM analysis.")
    st.stop()

tickers = [BANK_TICKERS[b] for b in selected_banks]

# -------- Date Range --------
start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.to_datetime("2018-01-01")
)

end_date = st.sidebar.date_input("End Date")

# -------- Lag Configuration --------
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

# -------- IRF Configuration --------
st.sidebar.subheader("IRF Configuration")

irf_horizon = st.sidebar.slider(
    "IRF Horizon (Periods)",
    min_value=5,
    max_value=20,
    value=10
)

# -------- Model Type --------
model_type = st.sidebar.radio(
    "Model Type",
    ["VAR", "VECM"]
)

# ---------------- Data ----------------
prices = get_price_data(tickers, start_date, end_date)
returns = get_returns(prices)

returns.columns = selected_banks

st.subheader("Daily Returns")
st.line_chart(returns)

# ======================================================
# ======================= VAR ==========================
# ======================================================
if model_type == "VAR":

    st.subheader("Vector Autoregression (VAR)")

    if returns.shape[1] < 2:
        st.warning("VAR requires at least two banks. Please select more banks.")
        st.stop()

    var_results = fit_var(
        returns,
        maxlags=max_lags,
        ic=lag_criterion
    )

    # ---- Model Info ----
    col1, col2 = st.columns(2)
    col1.metric("Selected Lag Order", var_results.k_ar)
    col2.metric(
        f"{lag_criterion.upper()}",
        round(getattr(var_results, lag_criterion), 2)
    )

    # ---- Stability Check ----
    st.markdown("### 🔍 Model Stability")
    if var_results.is_stable(verbose=False):
        st.success("VAR model is stable (all eigenvalues lie within the unit circle).")
    else:
        st.warning("VAR model may be unstable. Interpret results with caution.")

    # ---- Summary ----
    with st.expander("📄 VAR Model Summary"):
        st.text(var_results.summary())

    # ---- Impulse Response ----
    st.subheader("Impulse Response Functions (IRF)")

    irf = var_results.irf(irf_horizon)
    fig = irf.plot(orth=False)
    st.pyplot(fig)

    st.markdown("""
    **Interpretation:**  
    Impulse Response Functions illustrate how a shock to one bank’s returns
    propagates to other banks over time, capturing short-run interdependencies
    in the banking system.
    """)


# ======================================================
# ======================= VECM =========================
# ======================================================
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

    # Cointegration rank at 5% significance
    coint_rank = sum(trace_stats > crit_95)

    st.success(
        f"Estimated Number of Cointegrating Relations (5% level): **{coint_rank}**"
    )

    # ---------------- VECM Fit ----------------
    if coint_rank == 0:
        st.warning(
            "No cointegration detected at the 5% level. "
            "VECM may not be appropriate for this set of banks."
        )
    else:
        vecm_results = fit_vecm(
            returns,
            coint_rank=coint_rank,
            k_ar_diff=max_lags
        )

        # ==================================================
        # =============== LONG-RUN RELATIONSHIP ============
        # ==================================================
        st.subheader("🔗 Long-Run Equilibrium (Cointegrating Vectors)")

        beta = pd.DataFrame(
            vecm_results.beta,
            index=returns.columns,
            columns=[f"Cointegrating Vector {i+1}" for i in range(coint_rank)]
        )

        st.dataframe(beta.round(4), use_container_width=True)

        st.markdown("""
        **Long-run interpretation:**  
        These coefficients define the **equilibrium relationship** binding the
        selected banks. Deviations from this relationship represent
        long-run disequilibrium.
        """)

        # ==================================================
        # =============== SHORT-RUN ADJUSTMENT ==============
        # ==================================================
        st.subheader("⚖️ Short-Run Adjustment (Error Correction Term)")

        alpha = pd.DataFrame(
            vecm_results.alpha,
            index=returns.columns,
            columns=[f"ECM {i+1}" for i in range(coint_rank)]
        )

        st.dataframe(alpha.round(4), use_container_width=True)

        st.markdown("""
        **Short-run interpretation:**  
        - Significant negative coefficients indicate **adjustment back toward
          long-run equilibrium**.  
        - Banks with weak or insignificant coefficients behave as
          **long-run leaders**, while others adjust.
        """)

        # ---------------- Full Summary ----------------
        with st.expander("📄 Full VECM Model Summary"):
            st.text(vecm_results.summary())
