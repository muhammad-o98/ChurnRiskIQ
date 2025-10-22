"""Optimized SHAP analyzer with caching and Plotly helpers."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import shap
import plotly.express as px
import streamlit as st


@st.cache_resource(show_spinner=False)
def get_tree_explainer(model):
    return shap.TreeExplainer(model)


@st.cache_data(show_spinner=False)
def compute_shap_values(explainer, X: np.ndarray, max_samples: int = 1000):
    if isinstance(X, pd.DataFrame):
        Xs = X.sample(min(len(X), max_samples), random_state=42)
    else:
        idx = np.random.RandomState(42).choice(np.arange(X.shape[0]), size=min(X.shape[0], max_samples), replace=False)
        Xs = X[idx]
    vals = explainer.shap_values(Xs)
    return vals, Xs


def to_dataframe(X: np.ndarray, feature_names: Optional[list] = None) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    if feature_names is None:
        feature_names = [f'f{i}' for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names)


def plotly_importance(shap_vals, X_df: pd.DataFrame, top_n: int = 20):
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals
    imp = np.abs(sv).mean(axis=0)
    imp_df = pd.DataFrame({'Feature': X_df.columns, 'Importance': imp}).sort_values('Importance', ascending=False).head(top_n)
    fig = px.bar(imp_df, x='Importance', y='Feature', orientation='h', title='Top Features', color='Importance', color_continuous_scale='Viridis')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    return fig, imp_df


def plotly_dependence(shap_vals, X_df: pd.DataFrame, feature: str, color_feature: Optional[str] = None):
    if isinstance(shap_vals, list):
        sv = shap_vals[1]
    else:
        sv = shap_vals
    y = sv[:, X_df.columns.get_loc(feature)]
    c = X_df[color_feature] if color_feature else None
    fig = px.scatter(x=X_df[feature], y=y, color=c, labels={'x': feature, 'y': f'SHAP({feature})', 'color': color_feature}, opacity=0.7)
    fig.update_traces(marker=dict(line=dict(width=0)))
    fig.update_layout(title=f'Dependence: {feature}')
    return fig


def top_contributors_for_row(shap_vals_row: np.ndarray, feature_names: list, k: int = 3):
    idx = np.argsort(np.abs(shap_vals_row))[::-1][:k]
    return [(feature_names[i], shap_vals_row[i]) for i in idx]
