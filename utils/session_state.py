"""
Session State Management for Streamlit App
Handles data persistence across pages
"""

import streamlit as st
import pandas as pd
import numpy as np
from typing import Any, Optional


def init_session_state():
    """Initialize all session state variables"""
    
    # Data-related states
    if 'data_raw' not in st.session_state:
        st.session_state.data_raw = None
    
    if 'data_processed' not in st.session_state:
        st.session_state.data_processed = None
    
    if 'X_train' not in st.session_state:
        st.session_state.X_train = None
    
    if 'X_test' not in st.session_state:
        st.session_state.X_test = None
    
    if 'y_train' not in st.session_state:
        st.session_state.y_train = None
    
    if 'y_test' not in st.session_state:
        st.session_state.y_test = None
    
    # Model-related states
    if 'trained_models' not in st.session_state:
        st.session_state.trained_models = {}
    
    if 'best_model' not in st.session_state:
        st.session_state.best_model = None
    
    if 'best_model_name' not in st.session_state:
        st.session_state.best_model_name = None
    
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = None
    
    if 'preprocessor' not in st.session_state:
        st.session_state.preprocessor = None
    
    # SHAP-related states
    if 'shap_values' not in st.session_state:
        st.session_state.shap_values = None
    
    if 'shap_explainer' not in st.session_state:
        st.session_state.shap_explainer = None
    
    if 'shap_feature_names' not in st.session_state:
        st.session_state.shap_feature_names = None
    
    if 'shap_importance_df' not in st.session_state:
        st.session_state.shap_importance_df = None
    
    # Risk segmentation states
    if 'risk_segments' not in st.session_state:
        st.session_state.risk_segments = None
    
    if 'high_risk_customers' not in st.session_state:
        st.session_state.high_risk_customers = None
    
    # UI states
    if 'current_page' not in st.session_state:
        st.session_state.current_page = 'home'
    
    if 'data_uploaded' not in st.session_state:
        st.session_state.data_uploaded = False
    
    if 'models_trained' not in st.session_state:
        st.session_state.models_trained = False
    
    if 'shap_computed' not in st.session_state:
        st.session_state.shap_computed = False


def reset_session_state():
    """Reset all session state variables"""
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    init_session_state()


def save_data(data: pd.DataFrame, key: str = 'data_processed'):
    """Save processed data to session state"""
    st.session_state[key] = data.copy()


def load_data(key: str = 'data_processed') -> Optional[pd.DataFrame]:
    """Load data from session state"""
    return st.session_state.get(key, None)


def save_model(model: Any, name: str):
    """Save a trained model to session state"""
    st.session_state.trained_models[name] = model


def load_model(name: str) -> Optional[Any]:
    """Load a trained model from session state"""
    return st.session_state.trained_models.get(name, None)


def save_best_model(model: Any, name: str):
    """Save the best performing model"""
    st.session_state.best_model = model
    st.session_state.best_model_name = name


def has_data() -> bool:
    """Check if data is loaded"""
    return st.session_state.data_processed is not None


def has_trained_models() -> bool:
    """Check if models are trained"""
    return len(st.session_state.trained_models) > 0


def has_shap_analysis() -> bool:
    """Check if SHAP analysis is completed"""
    return st.session_state.shap_values is not None


def get_workflow_status():
    """Get current workflow completion status"""
    return {
        'data_uploaded': st.session_state.data_uploaded,
        'models_trained': st.session_state.models_trained,
        'shap_computed': st.session_state.shap_computed,
        'has_data': has_data(),
        'has_models': has_trained_models(),
        'has_shap': has_shap_analysis()
    }
