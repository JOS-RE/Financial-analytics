import streamlit as st

if st.session_state.get("mode") not in ["advanced"]:
    st.stop()
