"""
Model Training and Evaluation Page
"""

import streamlit as st

st.set_page_config(page_title="Model Training", page_icon="🤖", layout="wide")

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import sys
sys.path.append('..')

from utils.session_state import init_session_state, save_model, save_best_model
from utils.data_utils import get_preprocessor
from utils.ui import apply_theme
from utils.model_utils import (
    get_model_registry, train_model, evaluate_model, 
    compare_models, get_best_model
)
from models.ensemble import train_advanced, AdvancedTrainConfig
from models.persistence import save_model_version, log_to_mlflow

# Initialize
init_session_state()
apply_theme()

st.markdown('<h1 style="color: #1E3A8A;">🤖 Model Training & Evaluation</h1>', unsafe_allow_html=True)

# Check if data is available
if not st.session_state.get('data_uploaded', False) or st.session_state.get('X_train') is None:
    st.warning("⚠️ Please upload and preprocess data first in the Data Upload page!")
    st.stop()

# Model selection
st.markdown("### 🎯 Select Models to Train")

registry = get_model_registry()

col1, col2, col3 = st.columns(3)

selected_models = []

with col1:
    if st.checkbox("Decision Tree", value=True):
        selected_models.append('dt')
    if st.checkbox("Gradient Boosting", value=True):
        selected_models.append('gb')

with col2:
    if st.checkbox("Random Forest", value=True):
        selected_models.append('rf')
    if st.checkbox("XGBoost", value=True):
        selected_models.append('xgb')

with col3:
    if st.checkbox("LightGBM", value=True):
        selected_models.append('lgbm')
    if st.checkbox("CatBoost", value=True):
        selected_models.append('cat')

st.markdown("---")

# Advanced options
with st.expander("⚙️ Advanced Training Options"):
    colA, colB, colC = st.columns(3)
    with colA:
        use_smote = st.checkbox("Handle imbalance (SMOTE)", value=True)
        smote_kind = st.selectbox("Resampler", ["SMOTE","ADASYN"], index=0)
        calibrate = st.checkbox("Calibrate probabilities", value=True)
    with colB:
        ensemble_voting = st.checkbox("Use Voting Ensemble", value=True)
        ensemble_stacking = st.checkbox("Use Stacking Ensemble", value=True)
        cv_splits = st.slider("CV folds", 3, 10, 5)
    with colC:
        use_optuna = st.checkbox("Hyperparameter Tuning (Optuna)", value=False)
        n_trials = st.slider("Optuna trials", 5, 50, 20)
        scoring = st.selectbox("Optimize for", ["pr_auc"], index=0)

# Training section
if len(selected_models) == 0:
    st.info("👆 Please select at least one model to train.")
else:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"**{len(selected_models)} model(s) selected:** {', '.join([registry[m]['name'] for m in selected_models])}")
    
    with col2:
        train_button = st.button("🚀 Train Models", type="primary", width="stretch")
    
    if train_button:
        # Get preprocessor
        if st.session_state.preprocessor is None:
            st.session_state.preprocessor = get_preprocessor(st.session_state.X_train)
        
        # Advanced training pipeline (ensembles, CV, calibration, optional Optuna & SMOTE)
        cfg = AdvancedTrainConfig(
            cv_splits=cv_splits,
            use_smote=use_smote,
            smote_kind=smote_kind,
            use_optuna=use_optuna,
            n_trials=n_trials,
            calibrate=calibrate,
            ensemble_voting=ensemble_voting,
            ensemble_stacking=ensemble_stacking,
            scoring=scoring,
        )
        best_pipe, results, comparison_df = train_advanced(
            selected_models,
            st.session_state.preprocessor,
            st.session_state.X_train,
            st.session_state.y_train,
            st.session_state.X_test,
            st.session_state.y_test,
            cfg,
        )
        st.session_state.trained_models['advanced_best'] = best_pipe
        st.session_state.model_metrics = results
        st.session_state.models_trained = True
        st.session_state.comparison_df = comparison_df.copy()

        # Persist best model with metadata
        top_row = comparison_df.iloc[0]
        best_name = str(top_row['Model'])
        st.session_state.best_model = best_pipe
        st.session_state.best_model_name = best_name
        meta = save_model_version(
            best_pipe,
            name=best_name,
            metrics={
                'accuracy': float(top_row['Accuracy']),
                'precision': float(top_row['Precision']),
                'recall': float(top_row['Recall']),
                'f1': float(top_row['F1-Score']),
                'roc_auc': float(top_row['ROC AUC']),
                'pr_auc': float(top_row['PR AUC']),
                'brier': float(top_row['Brier Score']),
            },
            params={'cv_splits': cv_splits, 'use_smote': use_smote, 'smote_kind': smote_kind, 'use_optuna': use_optuna, 'n_trials': n_trials, 'calibrate': calibrate, 'ensemble_voting': ensemble_voting, 'ensemble_stacking': ensemble_stacking},
            notes='Advanced training run',
        )
        log_to_mlflow('Churn_Models', meta.metrics, meta.params)

        st.success(f"🏆 Best Model: **{best_name}** (PR AUC: {float(top_row['PR AUC']):.4f})")
        st.balloons()

# Show results if models are trained
if st.session_state.models_trained and st.session_state.model_metrics is not None:
    st.markdown("---")
    st.markdown("### 📊 Model Comparison")
    
    # Create comparison DataFrame (use advanced comparison if present)
    if 'comparison_df' in st.session_state and st.session_state.comparison_df is not None:
        comparison_df = st.session_state.comparison_df
    else:
        comparison_df = compare_models(st.session_state.model_metrics)
        st.session_state.comparison_df = comparison_df.copy()
    
    # Format for display
    display_df = comparison_df.copy()
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC', 'Brier Score']:
        display_df[col] = display_df[col].apply(lambda x: f"{x:.4f}")
    
    # Highlight best model
    def highlight_best(row):
        if row.name == 0:  # First row (best by PR AUC)
            return ['background-color: #D1FAE5; font-weight: bold'] * len(row)
        return [''] * len(row)
    
    styled_df = display_df.style.apply(highlight_best, axis=1)
    st.dataframe(styled_df, width="stretch")
    
    # Download results
    csv = comparison_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download Results",
        data=csv,
        file_name="model_comparison.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    # Visualizations
    st.markdown("### 📈 Performance Visualizations")
    
    tab1, tab2, tab3 = st.tabs(["📊 Metrics Comparison", "🎯 ROC Curves", "📉 Detailed Metrics"])
    
    with tab1:
        # Bar chart comparison
        metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC', 'PR AUC']
        
        fig = go.Figure()
        
        for metric in metrics_to_plot:
            fig.add_trace(go.Bar(
                name=metric,
                x=comparison_df['Model'],
                y=comparison_df[metric],
                text=comparison_df[metric].apply(lambda x: f"{x:.3f}"),
                textposition='auto'
            ))
        
        fig.update_layout(
            title="Model Performance Comparison",
            xaxis_title="Model",
            yaxis_title="Score",
            barmode='group',
            height=500
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with tab2:
        # ROC curves
        from sklearn.metrics import roc_curve
        
        fig = go.Figure()
        
        for model_key, results in st.session_state.model_metrics.items():
            y_test_probs = results['y_test_probs']
            fpr, tpr, _ = roc_curve(st.session_state.y_test, y_test_probs)
            
            fig.add_trace(go.Scatter(
                x=fpr,
                y=tpr,
                name=f"{registry[model_key]['name']} (AUC={results['test_metrics']['roc_auc']:.3f})",
                mode='lines'
            ))
        
        # Add diagonal
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            name='Random',
            mode='lines',
            line=dict(dash='dash', color='gray')
        ))
        
        fig.update_layout(
            title="ROC Curves",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=500
        )
        
        st.plotly_chart(fig, width="stretch")
    
    with tab3:
        # Select model for detailed view
        model_options = {registry[k]['name']: k for k in st.session_state.model_metrics.keys()}
        selected_model_name = st.selectbox("Select Model", list(model_options.keys()))
        selected_model_key = model_options[selected_model_name]
        
        results = st.session_state.model_metrics[selected_model_key]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📊 Test Set Metrics")
            test_metrics = results['test_metrics']
            metrics_df = pd.DataFrame({
                'Metric': list(test_metrics.keys()),
                'Value': [f"{v:.4f}" for v in test_metrics.values()]
            })
            st.table(metrics_df)
        
        with col2:
            st.markdown("#### 🎯 Confusion Matrix")
            cm = results['confusion_matrix']
            
            fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Predicted No Churn', 'Predicted Churn'],
                y=['Actual No Churn', 'Actual Churn'],
                text=cm,
                texttemplate='%{text}',
                textfont={"size": 16},
                colorscale='Blues'
            ))
            
            fig.update_layout(
                title="Confusion Matrix",
                height=400
            )
            
            st.plotly_chart(fig, width="stretch")
        
        # Best threshold
        st.info(f"💡 **Optimal Threshold:** {results['best_threshold']:.4f} (based on F1 score)")

# Navigation hint
if st.session_state.models_trained:
    st.markdown("---")
    st.markdown("""
    <div style="background-color: #D1FAE5; border-left: 4px solid #10B981; padding: 1rem; border-radius: 5px;">
        <h3 style="margin-top: 0; color: #1F2937;">✅ Models Trained!</h3>
        <p style="color: #1F2937;">Your models are ready for analysis. Head to the <strong>SHAP Analysis</strong> page to understand feature importance!</p>
    </div>
    """, unsafe_allow_html=True)
