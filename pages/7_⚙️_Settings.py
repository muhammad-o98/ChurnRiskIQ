"""
Settings Page - Preferences and Model Management
"""
import streamlit as st
import json
from pathlib import Path
import sys
sys.path.append('..')

from utils.session_state import init_session_state
from utils.cache_manager import set_pref, get_pref
from models.persistence import latest_best_model
from utils.ui import apply_theme

st.set_page_config(page_title="Settings", page_icon="⚙️", layout="wide")
init_session_state()
apply_theme()

st.markdown('<h1 style="color: #1E3A8A;">⚙️ Settings</h1>', unsafe_allow_html=True)

# Preferences
st.markdown("### User Preferences")
ms = st.number_input("Max SHAP samples", 100, 5000, value=get_pref('max_shap_samples', 1000))
th = st.slider("Alert PR-AUC threshold", 0.5, 0.9, value=float(get_pref('perf_threshold', 0.65)))
set_pref('max_shap_samples', ms)
set_pref('perf_threshold', th)

# Model registry
st.markdown("### Model Registry")
info = latest_best_model()
if info:
    st.json(info)
else:
    st.info("No models saved yet.")
