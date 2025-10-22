"""
Telecom Churn Prediction - Streamlit Application
Complete end-to-end machine learning app with SHAP analysis and risk segmentation
"""

import streamlit as st
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Telecom Churn Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern look
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        padding: 1rem 0;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2563EB;
        margin: 1.5rem 0 1rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .info-box {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1E3A8A 0%, #3B82F6 100%);
    }
    div[data-testid="stSidebar"] .css-1d391kg {
        color: white;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
</style>
""", unsafe_allow_html=True)

# Main page content
st.markdown('<h1 class="main-header">📊 Telecom Churn Analytics Platform</h1>', unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
    <h3>🎯 Welcome to the Advanced Churn Analytics Platform</h3>
    <p>This application provides end-to-end machine learning capabilities for predicting and analyzing customer churn in telecom services.</p>
</div>
""", unsafe_allow_html=True)

# Navigation
st.sidebar.title("🧭 Navigation")
st.sidebar.markdown("---")

# Feature highlights
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2.5rem;">📈</h2>
        <h4>Dashboard</h4>
        <p style="font-size: 0.9rem;">Key metrics & insights</p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2.5rem;">🤖</h2>
        <h4>Model Training</h4>
        <p style="font-size: 0.9rem;">Train & evaluate models</p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2.5rem;">🔍</h2>
        <h4>SHAP Analysis</h4>
        <p style="font-size: 0.9rem;">Feature importance</p>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <div style="background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); padding: 1.5rem; border-radius: 10px; color: white; text-align: center;">
        <h2 style="margin: 0; font-size: 2.5rem;">⚠️</h2>
        <h4>Risk Segments</h4>
        <p style="font-size: 0.9rem;">Customer segmentation</p>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Getting Started Section
st.markdown('<h2 class="sub-header">🚀 Getting Started</h2>', unsafe_allow_html=True)

st.markdown("""
### How to use this application:

1. **📤 Upload Data** (Data Upload page)
   - Upload your customer dataset in CSV format
   - The app will automatically preprocess and engineer features
   
2. **🤖 Train Models** (Model Training page)
   - Select from 6 different machine learning algorithms
   - Compare models using multiple metrics (ROC AUC, PR AUC, etc.)
   - Visualize performance with interactive charts

3. **🔍 Analyze with SHAP** (SHAP Analysis page)
   - Understand which features drive churn predictions
   - Explore feature importance globally and per-customer
   - Generate professional visualizations

4. **⚠️ Identify High-Risk Customers** (Risk Segmentation page)
   - Segment customers into High/Medium/Low risk categories
   - Get actionable retention strategies
   - Export customer lists for targeted campaigns

5. **📊 Monitor Dashboard** (Dashboard page)
   - View key metrics and KPIs
   - Track model performance
   - Generate executive reports
""")

st.markdown("---")

# Quick Stats (placeholder for demo)
st.markdown('<h2 class="sub-header">📈 Quick Overview</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.metric("Total Customers", "7,043", delta="↑ 2.1%")

with col2:
    st.metric("Churn Rate", "26.5%", delta="↓ 1.2%", delta_color="inverse")

with col3:
    st.metric("Model Accuracy", "84.2%", delta="↑ 3.5%")

with col4:
    st.metric("High Risk", "892", delta="↑ 45")

with col5:
    st.metric("ROI from Model", "$1.2M", delta="↑ $180K")

st.markdown("---")

# System Information
with st.expander("ℹ️ System Information & Requirements"):
    st.markdown("""
    **Supported Models:**
    - Decision Tree (DT)
    - Random Forest (RF)
    - Gradient Boosting (GB)
    - XGBoost (XGB)
    - LightGBM (LGBM)
    - CatBoost (CAT)
    
    **Key Features:**
    - ✅ Automatic feature engineering
    - ✅ SHAP explanations
    - ✅ Risk segmentation
    - ✅ Session state persistence
    - ✅ Export capabilities
    - ✅ Interactive visualizations
    
    **Data Requirements:**
    - CSV format with customer churn data
    - Minimum fields: tenure, MonthlyCharges, Contract, etc.
    - See example dataset in `/data/churndata.csv`
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; padding: 2rem 0;">
    <p>Built with ❤️ using Streamlit | © 2025 Telecom Churn Analytics</p>
</div>
""", unsafe_allow_html=True)
