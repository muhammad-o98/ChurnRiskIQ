"""
SHAP Analysis Page - Feature Importance and Explainability
"""

import streamlit as st

st.set_page_config(page_title="SHAP Analysis", page_icon="🔍", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import sys
sys.path.append('..')

from utils.session_state import init_session_state
from utils.data_utils import get_feature_names_from_preprocessor

init_session_state()

st.markdown('<h1 style="color: #1E3A8A;">🔍 SHAP Feature Importance Analysis</h1>', unsafe_allow_html=True)

# Check prerequisites
if not st.session_state.get('models_trained', False) or st.session_state.get('best_model') is None:
    st.warning("⚠️ Please train models first in the Model Training page!")
    st.stop()

# Compute SHAP values
if st.session_state.shap_values is None:
    if st.button("🚀 Compute SHAP Values", type="primary"):
        with st.spinner("Computing SHAP values... This may take a minute."):
            # Extract components
            preprocessor = st.session_state.best_model.named_steps['preprocessor']
            classifier = st.session_state.best_model.named_steps['classifier']
            X_test_transformed = preprocessor.transform(st.session_state.X_test)
            
            # Get feature names
            feature_names = get_feature_names_from_preprocessor(preprocessor, st.session_state.X_train)
            st.session_state.shap_feature_names = feature_names
            
            # Compute SHAP
            try:
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_test_transformed)
                st.session_state.shap_explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.shap_computed = True
                st.success("✅ SHAP values computed!")
                st.rerun()
            except Exception as e:
                st.error(f"TreeExplainer failed, using KernelExplainer (slower): {str(e)}")
                background = shap.sample(X_test_transformed, min(50, X_test_transformed.shape[0]))
                explainer = shap.KernelExplainer(lambda x: classifier.predict_proba(x)[:, 1], background)
                shap_values = explainer.shap_values(X_test_transformed, nsamples=100)
                st.session_state.shap_explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.shap_computed = True
                st.success("✅ SHAP values computed!")
                st.rerun()

# Display SHAP analysis if computed
if st.session_state.shap_computed:
    shap_values = st.session_state.shap_values
    feature_names = st.session_state.shap_feature_names
    X_test_transformed = st.session_state.best_model.named_steps['preprocessor'].transform(st.session_state.X_test)
    
    # Handle multi-output
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    
    # Compute importance
    mean_abs_shap = np.abs(sv).mean(axis=0)
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': mean_abs_shap
    }).sort_values('Importance', ascending=False)
    st.session_state.shap_importance_df = importance_df
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Summary Plot", "📈 Feature Importance", "🔍 Top Drivers"])
    
    with tab1:
        st.markdown("### SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_test_transformed, feature_names=feature_names, show=False, max_display=20)
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### SHAP Feature Importance")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_test_transformed, feature_names=feature_names, plot_type='bar', show=False, max_display=15)
        st.pyplot(fig)
    
    with tab3:
        st.markdown("### 🔍 Top 10 Churn Drivers")
        top_10 = importance_df.head(10).reset_index(drop=True)
        top_10.index = top_10.index + 1
        st.dataframe(top_10.style.format({'Importance': '{:.6f}'}), width="stretch")
        
        csv = top_10.to_csv().encode('utf-8')
        st.download_button("📥 Download Top Drivers", csv, "top_churn_drivers.csv", "text/csv")
