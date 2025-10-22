"""
Dashboard - Executive Summary
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('..')

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

from utils.session_state import init_session_state, has_data
from utils.ui import apply_theme, header

init_session_state()
apply_theme()

header("Executive Dashboard", "📊")

# Check prerequisites (robust to older/newer keys)
if not has_data():
    st.warning("⚠️ Please upload and preprocess data first on the Data Upload page!")
    st.stop()

# Overview section
st.markdown("## 📋 Project Overview")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total = len(st.session_state.get('data_processed', pd.DataFrame()))
    st.metric("Total Customers", f"{total:,}" if total > 0 else "No data")

with col2:
    df = st.session_state.get('data_processed')
    if df is not None and 'Churn' in df.columns:
        churn_rate = df['Churn'].mean() * 100
        st.metric("Overall Churn Rate", f"{churn_rate:.1f}%")
    else:
        st.metric("Overall Churn Rate", "N/A")

with col3:
    if st.session_state.get('models_trained', False):
        trained_models = st.session_state.get('trained_models', {})
        st.metric("Models Trained", len(trained_models))
    else:
        st.metric("Models Trained", "0")

with col4:
    best_model_name = st.session_state.get('best_model_name')
    if best_model_name:
        st.metric("Best Model", best_model_name)
    else:
        st.metric("Best Model", "Not Selected")

st.markdown("---")

# Model Performance
if st.session_state.get('models_trained', False):
    st.markdown("## 🤖 Model Performance Summary")
    
    # Comparison table
    comparison_df = st.session_state.get('comparison_df')
    if comparison_df is not None:
        st.dataframe(
            comparison_df.style.background_gradient(cmap='RdYlGn', subset=['Accuracy', 'PR AUC', 'F1-Score']),
            width="stretch"
        )
        
        # Best model metrics
        col1, col2 = st.columns(2)
        
        with col1:
            best_model_name = st.session_state.get('best_model_name')
            if best_model_name:
                best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
                
                st.markdown(f"### 🏆 Best Model: {best_model_name}")
                
                metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
                with metrics_col1:
                    st.metric("Accuracy", f"{best_row['Accuracy']:.4f}")
                with metrics_col2:
                    st.metric("PR AUC", f"{best_row['PR AUC']:.4f}")
                with metrics_col3:
                    st.metric("F1-Score", f"{best_row['F1-Score']:.4f}")
        
        with col2:
            # Model comparison chart
            fig = px.bar(
                comparison_df,
                x='Model',
                y=['Accuracy', 'PR AUC', 'F1-Score'],
                title='Model Comparison',
                barmode='group',
                color_discrete_sequence=['#3B82F6', '#10B981', '#F59E0B']
            )
            st.plotly_chart(fig, width="stretch")

# Risk Segmentation
risk_segments = st.session_state.get('risk_segments')
if risk_segments is not None:
    st.markdown("---")
    st.markdown("## ⚠️ Risk Distribution")
    
    X_full = risk_segments
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Risk distribution pie chart
        fig = px.pie(
            X_full,
            names='risk_segment',
            title='Customer Risk Distribution',
            color='risk_segment',
            color_discrete_map={'High Risk': '#EF4444', 'Medium Risk': '#F59E0B', 'Low Risk': '#10B981'},
            hole=0.4
        )
        st.plotly_chart(fig, width="stretch")
    
    with col2:
        # Risk vs actual churn
        risk_summary = X_full.groupby('risk_segment').agg({
            'churn_probability': 'count',
            'actual_churn': 'sum'
        }).reset_index()
        risk_summary.columns = ['Risk Segment', 'Total Customers', 'Actual Churned']
        risk_summary['Churn Rate (%)'] = (risk_summary['Actual Churned'] / risk_summary['Total Customers'] * 100).round(2)
        
        fig = px.bar(
            risk_summary,
            x='Risk Segment',
            y=['Total Customers', 'Actual Churned'],
            title='Risk Segment Analysis',
            barmode='group',
            color_discrete_sequence=['#3B82F6', '#EF4444']
        )
        st.plotly_chart(fig, width="stretch")
    
    # Key insights
    st.markdown("### 🎯 Key Insights")
    
    high_risk_count = (X_full['risk_segment'] == 'High Risk').sum()
    high_risk_pct = high_risk_count / len(X_full) * 100
    high_risk_churn = X_full[X_full['risk_segment'] == 'High Risk']['actual_churn'].mean() * 100
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div style="background-color: #FEE2E2; padding: 1rem; border-radius: 10px;">
            <h4 style="color: #991B1B;">🚨 High Risk Alert</h4>
            <p><strong>{high_risk_count:,} customers</strong> ({high_risk_pct:.1f}%)</p>
            <p>Actual churn rate: <strong>{high_risk_churn:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        medium_risk_count = (X_full['risk_segment'] == 'Medium Risk').sum()
        medium_risk_pct = medium_risk_count / len(X_full) * 100
        medium_risk_churn = X_full[X_full['risk_segment'] == 'Medium Risk']['actual_churn'].mean() * 100
        
        st.markdown(f"""
        <div style="background-color: #FEF3C7; padding: 1rem; border-radius: 10px;">
            <h4 style="color: #92400E;">⚡ Medium Risk</h4>
            <p><strong>{medium_risk_count:,} customers</strong> ({medium_risk_pct:.1f}%)</p>
            <p>Actual churn rate: <strong>{medium_risk_churn:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        low_risk_count = (X_full['risk_segment'] == 'Low Risk').sum()
        low_risk_pct = low_risk_count / len(X_full) * 100
        low_risk_churn = X_full[X_full['risk_segment'] == 'Low Risk']['actual_churn'].mean() * 100
        
        st.markdown(f"""
        <div style="background-color: #D1FAE5; padding: 1rem; border-radius: 10px;">
            <h4 style="color: #065F46;">✅ Low Risk</h4>
            <p><strong>{low_risk_count:,} customers</strong> ({low_risk_pct:.1f}%)</p>
            <p>Actual churn rate: <strong>{low_risk_churn:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

# Feature Importance
# Feature Importance
shap_values = st.session_state.get('shap_values')
if shap_values is not None:
    st.markdown("---")
    st.markdown("## 🔍 Top Churn Drivers")
    
    import shap
    import numpy as np
    
    # Calculate mean absolute SHAP values with feature names from session
    feature_names = st.session_state.get('shap_feature_names')
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(np.array(shap_values[1] if isinstance(shap_values, list) else shap_values).shape[1])]

    if isinstance(shap_values, list):
        shap_abs = np.abs(shap_values[1])
    else:
        shap_abs = np.abs(shap_values)

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'Importance': shap_abs.mean(axis=0)
    }).sort_values('Importance', ascending=False).head(10)

    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='Top 10 Churn Drivers',
        color='Importance',
        color_continuous_scale='Reds'
    )
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, width="stretch")

# Monitoring & Confidence Distribution
st.markdown("---")
st.markdown("## 🧭 Monitoring")

# Performance alert
perf_threshold = float(st.session_state.get('prefs', {}).get('perf_threshold', 0.65))
comp = st.session_state.get('comparison_df')
if comp is not None and not comp.empty:
    top_pr = float(comp.sort_values('PR AUC', ascending=False).iloc[0]['PR AUC'])
    if top_pr < perf_threshold:
        st.warning(f"⚠️ Model PR-AUC {top_pr:.3f} below threshold {perf_threshold:.3f}. Consider retraining.")
    else:
        st.success(f"✅ Model PR-AUC {top_pr:.3f} meets threshold {perf_threshold:.3f}.")

# Confidence distribution
bm = st.session_state.get('best_model')
if bm is not None and st.session_state.get('X_test') is not None:
    try:
        probs = bm.predict_proba(st.session_state.X_test)[:, 1]
        figc = px.histogram(probs, nbins=30, title='Prediction Confidence Distribution', color_discrete_sequence=['#2563EB'])
        st.plotly_chart(figc, use_container_width=True)
    except Exception:
        pass

# Executive Summary
st.markdown("---")
st.markdown("## 📝 Executive Summary")

df = st.session_state.get('data_processed')
best_model_name = st.session_state.get('best_model_name')
comparison_df = st.session_state.get('comparison_df')

summary_text = "### Key Findings\n\n**Dataset Overview**\n"

if df is not None:
    summary_text += f"- Total customers analyzed: **{len(df):,}**\n"
    
    if 'Churn' in df.columns:
        churn_rate = df['Churn'].mean() * 100
        summary_text += f"- Overall churn rate: **{churn_rate:.1f}%**\n"

if best_model_name and comparison_df is not None:
    best_row = comparison_df[comparison_df['Model'] == best_model_name].iloc[0]
    summary_text += f"""
**Model Performance**
- Best performing model: **{best_model_name}**
- Model accuracy: **{best_row['Accuracy']:.2%}**
- PR AUC score: **{best_row['PR AUC']:.4f}**
"""

if risk_segments is not None:
    summary_text += f"""
**Risk Segmentation**
- High risk customers: **{high_risk_count:,}** ({high_risk_pct:.1f}%) - Actual churn: {high_risk_churn:.1f}%
- Medium risk customers: **{medium_risk_count:,}** ({medium_risk_pct:.1f}%) - Actual churn: {medium_risk_churn:.1f}%
- Low risk customers: **{low_risk_count:,}** ({low_risk_pct:.1f}%) - Actual churn: {low_risk_churn:.1f}%

**Recommendations**
1. **Immediate Action**: Focus retention efforts on {high_risk_count:,} high-risk customers
2. **Proactive Monitoring**: Implement early warning systems for medium-risk segment
3. **Loyalty Programs**: Strengthen engagement with low-risk customers through VIP programs
4. **Key Drivers**: Address top churn factors identified in SHAP analysis
"""

st.markdown(summary_text)

# Export report
st.markdown("---")
st.markdown("### 📥 Export Report")

col1, col2 = st.columns(2)

with col1:
    if st.session_state.get('models_trained', False) and comparison_df is not None:
        csv = comparison_df.to_csv(index=False).encode('utf-8')
        st.download_button("📊 Model Comparison Report", csv, "model_comparison.csv", "text/csv")

with col2:
    if risk_segments is not None:
        summary_df = X_full.groupby('risk_segment').agg({
            'churn_probability': ['count', 'mean'],
            'actual_churn': ['sum', 'mean']
        })
        summary_df.columns = ['Customer Count', 'Avg Risk Score', 'Churned', 'Churn Rate']
        csv = summary_df.to_csv().encode('utf-8')
        st.download_button("⚠️ Risk Segmentation Report", csv, "risk_segmentation.csv", "text/csv")
