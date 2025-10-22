"""
Predictions Page - Single and Batch
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
sys.path.append('..')

from utils.session_state import init_session_state
from utils.data_utils import preprocess_data, engineer_features
from utils.validators import CustomerFeatures
from analysis.shap_analyzer import get_tree_explainer, compute_shap_values, to_dataframe, top_contributors_for_row
from analysis.risk_segmentation import recommended_interventions
from utils.ui import apply_theme

st.set_page_config(page_title="Predictions", page_icon="🔮", layout="wide")
init_session_state()
apply_theme()

st.markdown('<h1 style="color: #1E3A8A;">🔮 Predictions</h1>', unsafe_allow_html=True)

if st.session_state.get('best_model') is None:
    st.warning("⚠️ Train or load a model first.")
    st.stop()

model = st.session_state.best_model

# Tabs for Single and Batch
single_tab, batch_tab = st.tabs(["🎯 Single Prediction", "📦 Batch Prediction"])

with single_tab:
    st.markdown("### 📋 Customer Features")
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            gender = st.selectbox('gender', ['Male', 'Female'])
            SeniorCitizen = st.selectbox('SeniorCitizen', [0,1])
            Partner = st.selectbox('Partner', [0,1])
            Dependents = st.selectbox('Dependents', [0,1])
            tenure = st.number_input('tenure', 0, 120, 12)
            PhoneService = st.selectbox('PhoneService', [0,1])
        with col2:
            MultipleLines = st.selectbox('MultipleLines', [0,1])
            InternetService = st.selectbox('InternetService', ['DSL','Fiber optic','No'])
            OnlineSecurity = st.selectbox('OnlineSecurity', [0,1])
            OnlineBackup = st.selectbox('OnlineBackup', [0,1])
            TechSupport = st.selectbox('TechSupport', [0,1])
            StreamingTV = st.selectbox('StreamingTV', [0,1])
        with col3:
            StreamingMovies = st.selectbox('StreamingMovies', [0,1])
            Contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
            PaperlessBilling = st.selectbox('PaperlessBilling', [0,1])
            PaymentMethod = st.selectbox('PaymentMethod', ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
            MonthlyCharges = st.number_input('MonthlyCharges', 0.0, 500.0, 70.0, step=0.1)
            TotalCharges = st.number_input('TotalCharges', 0.0, 10000.0, 2000.0, step=1.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        try:
            cf = CustomerFeatures(
                gender=gender, SeniorCitizen=SeniorCitizen, Partner=Partner, Dependents=Dependents,
                tenure=tenure, PhoneService=PhoneService, MultipleLines=MultipleLines, InternetService=InternetService,
                OnlineSecurity=OnlineSecurity, OnlineBackup=OnlineBackup, TechSupport=TechSupport,
                StreamingTV=StreamingTV, StreamingMovies=StreamingMovies, Contract=Contract,
                PaperlessBilling=PaperlessBilling, PaymentMethod=PaymentMethod, MonthlyCharges=MonthlyCharges,
                TotalCharges=TotalCharges
            )
            df = pd.DataFrame([cf.model_dump()])
            df_proc = engineer_features(preprocess_data(df))
            proba = model.predict_proba(df_proc)[:, 1][0]
            st.metric("Churn Probability", f"{proba:.3f}")

            # SHAP explanation for the row (best-effort: TreeExplainer only if supported)
            try:
                clf = model.named_steps['classifier']
                pre = model.named_steps['preprocessor']
                X_t = pre.transform(df_proc)
                expl = get_tree_explainer(clf)
                shap_vals = expl.shap_values(X_t)
                sv_row = shap_vals[1][0] if isinstance(shap_vals, list) else shap_vals[0]
                feat_names = st.session_state.get('shap_feature_names')
                if not feat_names:
                    feat_names = [f'f{i}' for i in range(len(sv_row))]
                top3 = top_contributors_for_row(sv_row, feat_names, k=3)
                st.markdown("#### Top 3 Risk Factors")
                for f, v in top3:
                    st.write(f"- {f}: {v:.4f}")
            except Exception as e:
                st.info(f"SHAP explanation unavailable: {e}")

            # Risk intervention suggestion
            from analysis.risk_segmentation import segment_by_probability, recommended_interventions
            risk = segment_by_probability(np.array([proba]))[0]
            st.markdown(f"#### Risk Category: **{risk}**")
            strat = recommended_interventions(risk)
            if strat:
                st.json(strat)
        except Exception as e:
            st.error(f"Validation or prediction error: {e}")

with batch_tab:
    st.markdown("### 📤 Upload CSV for Batch Predictions")
    up = st.file_uploader("CSV with customer rows", type=['csv'])
    if up is not None:
        try:
            raw = pd.read_csv(up)
            st.write(f"Loaded {len(raw):,} rows")
            proc = engineer_features(preprocess_data(raw))
            probs = model.predict_proba(proc)[:, 1]

            # Per-row top factors (best-effort, sample for speed)
            topk = []
            try:
                clf = model.named_steps['classifier']
                pre = model.named_steps['preprocessor']
                Xt = pre.transform(proc)
                expl = get_tree_explainer(clf)
                shap_vals = expl.shap_values(Xt)
                sv = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
                feat_names = st.session_state.get('shap_feature_names')
                if feat_names is None:
                    feat_names = [f'f{i}' for i in range(sv.shape[1])]
                for i in range(min(2000, sv.shape[0])):  # cap to avoid huge payloads
                    idx = np.argsort(np.abs(sv[i]))[::-1][:3]
                    topk.append([feat_names[j] for j in idx])
                while len(topk) < len(proc):
                    topk.append([])
            except Exception:
                topk = [[] for _ in range(len(proc))]

            # Risk category + recommended intervention
            from analysis.risk_segmentation import segment_by_probability
            risk = segment_by_probability(probs)

            out = pd.DataFrame({
                'customer_index': proc.index,
                'churn_probability': probs,
                'risk_category': risk,
                'top_factors': [', '.join(x) for x in topk],
            })
            st.dataframe(out.head(50))
            st.download_button("📥 Download Results", out.to_csv(index=False).encode('utf-8'), "batch_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Batch prediction failed: {e}")
