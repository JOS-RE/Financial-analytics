import streamlit as st
from utils.constants import BANK_TICKERS

# ==================================================
# =============== PAGE CONFIG ======================
# ==================================================
st.set_page_config(
    page_title="FINA | Financial Intelligence & Analytics",
    layout="wide"
)

st.sidebar.image("assets/NMIMS_B.png", use_container_width=True)
st.sidebar.markdown("---")

# ==================================================
# =============== HIDE DEFAULT NAV =================
# ==================================================
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
# ================= BRANDING =======================
# ==================================================

col1,col1a,  col2,col3a, col3 = st.columns([1,1, 2,1, 1])

with col2:
    st.image("assets/NMIMS_B.png", use_container_width=True)


# ---- Banner ----
st.image("assets/Banner.png", use_container_width=True)

# ---- Logo + Title ----
col_logo, col_title = st.columns([1, 5])

# with col_logo:
# st.image("assets/NMIMS_B.png", width=120)

st.set_page_config(
    page_title="Financial Analytics Platform",
    layout="wide"
)
st.title("⭕ FINA - A Financial Analytics Platform")
st.caption("An integrated platform for portfolio construction, trading, and financial econometrics")




st.markdown(
    "An integrated platform for portfolio construction, volatility-aware trading, "
    "and financial econometrics."
)

st.divider()

# ==================================================
# =============== BANK SELECTION ===================
# ==================================================
st.subheader("Step 1: Select Companies")

selected_banks = st.multiselect(
    "Choose companies for analysis",
    options=list(BANK_TICKERS.keys()),
    default=st.session_state.banks,
    key="bank_selector"
)

st.session_state.banks = selected_banks

if st.session_state.banks:
    st.success(f"🟠 {len(st.session_state.banks)} companies(s) selected")

st.divider()

# ==================================================
# =============== OBJECTIVE SELECTION ==============
# ==================================================
st.subheader("Step 2: Choose Objective")

col1, col2, col3 = st.columns(3)

# ---------------- Long-Term ----------------
with col1:
    st.markdown("### 🟠 Long-Term Investing")
    st.caption("Portfolio optimisation & asset allocation")

    if st.button("📈 Go to Portfolio Optimisation", use_container_width=True):
        if not st.session_state.banks:
            st.warning("Please select at least one Org.")
        else:
            st.session_state.mode = "long_term"
            st.switch_page("pages/2_Portfolio_Optimisation.py")

# ---------------- Trading ----------------
with col2:
    st.markdown("### 🔴 Active Trading")
    st.caption("Volatility → GARCH → Trading signals")

    if st.button("🤖 Go to Trading", use_container_width=True):
        if not st.session_state.banks:
            st.warning("Please select at least one Org.")
        else:
            st.session_state.mode = "trading"
            st.switch_page("pages/4_GARCH_Volatility.py")

# ---------------- Advanced ----------------
with col3:
    st.markdown("### 🔴 Advanced Analytics")
    st.caption("VAR, VECM, DLM & system dynamics")

    if st.button("🔬 Go to Advanced Analysis", use_container_width=True):
        if len(st.session_state.banks) < 2:
            st.warning("Please select at least two Orgs.")
        else:
            st.session_state.mode = "advanced"
            st.switch_page("pages/3_VAR_VECM.py")

# ==================================================
# ================= STATUS =========================
# ==================================================
if st.session_state.banks:
    st.info(f"**Selected Companies:** {', '.join(st.session_state.banks)}")

# ==================================================
# ================= FOOTER =========================
# ==================================================
# st.markdown("---")

# if st.session_state.banks:
#     st.info(f"**Selected Companies:** {', '.join(st.session_state.banks)}")

st.markdown("""
---
### 🧭 How to use this platform
- **Long-Term Investing** → Portfolio Optimisation  
- **Active Trading** → Volatility modelling → Algorithmic strategies  
- **Advanced Analytics** → VAR, VECM & Distributed Lag Models  

Your selected Companies persist across modules.
""")

st.markdown("---")

# import streamlit as st
import pandas as pd

st.subheader("👥 Team – FINA")

team_df = pd.DataFrame([
    ["Joshith Reddy Gopidi", "joshithreedy.gopidi837@nmims.in"],
    ["Kavya T", "kavya.t701@nmims.in"],
    ["Sidharth Prakash", "sidharth.prakash383@nmims.in"],
    ["Sabarimayurnath U", "sabarimayurnath.u139@nmims.in"],
    ["Narendhran", "narendhran.171@nmims.in"],
    ["Vaishnavi Rajkumar", "vaishnavi.rajkumar320@nmims.in"],
], columns=["Name", "Email"])

st.table(team_df)


st.markdown(
    """

    **Programme:** MBA  
    **Institution:** SVKM’s Narsee Monjee Institute of Management Studies (NMIMS)  
    **Project:** Financial Analytics Capstone – **FINA**
    """
)
