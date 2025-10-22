"""
SHAP Analysis Page - Feature Importance and Explainability
"""

import streamlit as st

st.set_page_config(page_title="SHAP Analysis", page_icon="🔍", layout="wide")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import plotly.express as px
from sklearn.cluster import KMeans
from analysis.shap_analyzer import plotly_importance, plotly_dependence
import sys
sys.path.append('..')

from utils.session_state import init_session_state
from utils.data_utils import get_feature_names_from_preprocessor
from utils.ui import apply_theme

init_session_state()
apply_theme()

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
            # Sample for web performance
            max_samples = int(st.session_state.get('prefs', {}).get('max_shap_samples', 1000))
            if X_test_transformed.shape[0] > max_samples:
                import numpy as _np
                idx = _np.random.RandomState(42).choice(_np.arange(X_test_transformed.shape[0]), size=max_samples, replace=False)
                X_shap = X_test_transformed[idx]
            else:
                X_shap = X_test_transformed
            
            # Get feature names
            feature_names = get_feature_names_from_preprocessor(preprocessor, st.session_state.X_train)
            st.session_state.shap_feature_names = feature_names
            
            # Compute SHAP
            try:
                explainer = shap.TreeExplainer(classifier)
                shap_values = explainer.shap_values(X_shap)
                st.session_state.shap_explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.X_shap_display = X_shap
                st.session_state.shap_computed = True
                st.success("✅ SHAP values computed!")
                st.rerun()
            except Exception as e:
                st.error(f"TreeExplainer failed, using KernelExplainer (slower): {str(e)}")
                background = shap.sample(X_shap, min(50, X_shap.shape[0]))
                explainer = shap.KernelExplainer(lambda x: classifier.predict_proba(x)[:, 1], background)
                shap_values = explainer.shap_values(X_shap, nsamples=100)
                st.session_state.shap_explainer = explainer
                st.session_state.shap_values = shap_values
                st.session_state.X_shap_display = X_shap
                st.session_state.shap_computed = True
                st.success("✅ SHAP values computed!")
                st.rerun()

# Display SHAP analysis if computed
if st.session_state.shap_computed:
    shap_values = st.session_state.shap_values
    feature_names = st.session_state.shap_feature_names
    X_disp = st.session_state.get('X_shap_display')
    if X_disp is None:
        X_disp = st.session_state.best_model.named_steps['preprocessor'].transform(st.session_state.X_test)
    # Build display DataFrames
    X_df = pd.DataFrame(X_disp, columns=feature_names)
    
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
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Summary Plot", "📈 Feature Importance", "🔍 Top Drivers", "🧭 4D Explorer", "📦 Distributions"])
    
    with tab1:
        st.markdown("### SHAP Summary Plot")
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(sv, X_df.values, feature_names=feature_names, show=False, max_display=20)
        st.pyplot(fig)
    
    with tab2:
        st.markdown("### SHAP Feature Importance (Interactive)")
        fig_imp, _ = plotly_importance(sv, X_df, top_n=15)
        st.plotly_chart(fig_imp, use_container_width=True)
    
    with tab3:
        st.markdown("### 🔍 Top 10 Churn Drivers")
        top_10 = importance_df.head(10).reset_index(drop=True)
        top_10.index = top_10.index + 1
        st.dataframe(top_10.style.format({'Importance': '{:.6f}'}), width="stretch")
        
        csv = top_10.to_csv().encode('utf-8')
        st.download_button("📥 Download Top Drivers", csv, "top_churn_drivers.csv", "text/csv")

    with tab4:
        st.markdown("### 🧭 Interactive 4D SHAP Explorer")
        st.caption("Explore relationships among top features in 3D with a 4th dimension encoded as color or size. Optionally overlay clusters.")

        # Select features
        top_features = importance_df['Feature'].head(12).tolist()
        colA, colB, colC = st.columns(3)
        with colA:
            fx = st.selectbox("X-axis", top_features, index=0)
        with colB:
            fy = st.selectbox("Y-axis", top_features, index=1)
        with colC:
            fz = st.selectbox("Z-axis", top_features, index=2)

        # 4th dimension
        mode = st.radio("4th dimension", ["Color by SHAP of", "Color by feature", "Size by SHAP of"], horizontal=True)
        f4 = st.selectbox("Select 4th feature", top_features, index=3)

        # Clustering options
        cc1, cc2, cc3 = st.columns([1,1,2])
        with cc1:
            do_cluster = st.checkbox("Show clusters", value=False)
        with cc2:
            k = st.slider("k (clusters)", 2, 8, 3)
        with cc3:
            color_scale = st.selectbox("Color scale", ["Viridis", "Turbo", "Plasma", "Cividis", "Inferno", "Magma"], index=0)

        # Build plotting frame
        shap_mat = sv  # shape: (n_samples, n_features)
        shap_df = pd.DataFrame(shap_mat, columns=feature_names)

        plot_df = pd.DataFrame({
            'x': X_df[fx],
            'y': X_df[fy],
            'z': X_df[fz],
        })

        if mode == "Color by SHAP of":
            plot_df['color'] = shap_df[f4]
            size_arg = None
            color_title = f"SHAP({f4})"
        elif mode == "Color by feature":
            plot_df['color'] = X_df[f4]
            size_arg = None
            color_title = f4
        else:  # Size by SHAP of
            plot_df['color'] = None
            size_arg = np.abs(shap_df[f4]) + 1e-6
            size_arg = 5 + 10 * (size_arg / size_arg.max())
            color_title = None

        # Optional clustering on selected axes
        cluster_labels = None
        if do_cluster:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            cluster_labels = km.fit_predict(plot_df[['x','y','z']])
            plot_df['cluster'] = cluster_labels.astype(str)

        # 3D scatter
        if size_arg is None:
            fig3d = px.scatter_3d(
                plot_df,
                x='x', y='y', z='z',
                color=None if do_cluster else ('color' if 'color' in plot_df else None),
                color_continuous_scale=color_scale,
                labels={'x': fx, 'y': fy, 'z': fz, 'color': color_title} if color_title else {'x': fx, 'y': fy, 'z': fz},
                opacity=0.8
            )
            if do_cluster:
                fig3d = px.scatter_3d(
                    plot_df,
                    x='x', y='y', z='z',
                    color='cluster',
                    labels={'x': fx, 'y': fy, 'z': fz, 'cluster': 'Cluster'},
                    opacity=0.85
                )
        else:
            fig3d = px.scatter_3d(
                plot_df,
                x='x', y='y', z='z',
                size=size_arg,
                color=None if do_cluster else None,
                labels={'x': fx, 'y': fy, 'z': fz},
                opacity=0.8
            )

        fig3d.update_traces(marker=dict(line=dict(width=0)))
        fig3d.update_layout(title=f"3D: {fx} vs {fy} vs {fz}" + (f" — {color_title}" if color_title else ""), height=700)
        st.plotly_chart(fig3d, use_container_width=True)

        # Small helper charts
        st.markdown("#### Distributions of selected features")
        c1, c2, c3, c4 = st.columns(4)
        for ax, fname in zip([c1,c2,c3,c4],[fx,fy,fz,f4]):
            with ax:
                hist = px.histogram(X_df, x=fname, nbins=30, title=fname, color_discrete_sequence=['#3B82F6'])
                hist.update_layout(margin=dict(l=0,r=0,t=30,b=0), height=250)
                st.plotly_chart(hist, use_container_width=True)

    with tab5:
        st.markdown("### 📦 More SHAP Plots")
        st.caption("Quickly explore pairwise dependence across top features.")
        top = importance_df['Feature'].head(6).tolist()
        f_primary = st.selectbox("Feature", top, index=0)
        f_color = st.selectbox("Color by", top, index=1)
        fig_dep = plotly_dependence(sv, X_df, f_primary, f_color)
        fig_dep.update_layout(height=500)
        st.plotly_chart(fig_dep, use_container_width=True)
