import streamlit as st
from utils.constants import BANK_TICKERS

# ==================================================
# =============== PAGE CONFIG ======================
# ==================================================
st.set_page_config(
    page_title="Financial Analytics Platform",
    layout="wide"
)

# ==================================================
# =============== OPTIONAL: HIDE DEFAULT NAV =======
# ==================================================
# This hides Streamlit's auto-generated page list
st.markdown("""
<style>
section[data-testid="stSidebarNav"] {
    display: none;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# =============== SESSION STATE ====================
# ==================================================
if "banks" not in st.session_state:
    st.session_state.banks = []

if "mode" not in st.session_state:
    st.session_state.mode = None

# ==================================================
# ================= HOME PAGE ======================
# ==================================================
st.title("📊 Financial Analytics Platform")
st.caption("An integrated platform for portfolio construction, trading, and financial econometrics")

# ==================================================
# =============== BANK SELECTION ===================
# ==================================================
st.subheader("Step 1: Select Banks")

selected_banks = st.multiselect(
    "Choose banks for analysis",
    options=list(BANK_TICKERS.keys()),
    default=st.session_state.banks,
    key="bank_selector"
)

# Always sync session state
st.session_state.banks = selected_banks

if st.session_state.banks:
    st.success(f"✅ {len(st.session_state.banks)} bank(s) selected")

st.divider()

# ==================================================
# =============== OBJECTIVE SELECTION ==============
# ==================================================
st.subheader("Step 2: Choose Objective")

col1, col2, col3 = st.columns(3)

# ---------------- Long-Term ----------------
with col1:
    st.markdown("### 🟢 Long-Term Investing")
    st.caption("Portfolio optimisation & asset allocation")

    if st.button("📈 Go to Portfolio Optimisation", use_container_width=True):
        if not st.session_state.banks:
            st.warning("Please select at least one bank.")
        else:
            st.session_state.mode = "long_term"
            st.switch_page("pages/2_Portfolio_Optimisation.py")

# ---------------- Trading ----------------
with col2:
    st.markdown("### 🔵 Active Trading")
    st.caption("Volatility → GARCH → Trading signals")

    if st.button("🤖 Go to Trading", use_container_width=True):
        if not st.session_state.banks:
            st.warning("Please select at least one bank.")
        else:
            st.session_state.mode = "trading"
            st.switch_page("pages/4_GARCH_Volatility.py")

# ---------------- Advanced ----------------
with col3:
    st.markdown("### 🟣 Advanced Analytics")
    st.caption("VAR, VECM, DLM & system dynamics")

    if st.button("🔬 Go to Advanced Analysis", use_container_width=True):
        if len(st.session_state.banks) < 2:
            st.warning("Please select at least two banks.")
        else:
            st.session_state.mode = "advanced"
            st.switch_page("pages/3_VAR_VECM.py")

# ==================================================
# ================= STATUS =========================
# ==================================================
if st.session_state.banks:
    st.info(f"**Selected Banks:** {', '.join(st.session_state.banks)}")

st.markdown("""
---
### 🧭 How to use this platform
- **Long-Term Investing** → Portfolio Optimisation  
- **Active Trading** → Volatility modelling → Algorithmic strategies  
- **Advanced Analytics** → VAR, VECM & Distributed Lag Models  

Your selected banks persist across modules.
""")
