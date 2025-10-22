"""Session caching helpers and preference storage."""
from __future__ import annotations

from typing import Any, Dict
import streamlit as st


def set_pref(key: str, value: Any):
    st.session_state.setdefault('prefs', {})
    st.session_state['prefs'][key] = value


def get_pref(key: str, default: Any = None) -> Any:
    return st.session_state.get('prefs', {}).get(key, default)


def cache_predictions(key: str, df_hash: str, preds):
    st.session_state.setdefault('pred_cache', {})
    st.session_state['pred_cache'][key] = {'hash': df_hash, 'preds': preds}


def get_cached_predictions(key: str, df_hash: str):
    item = st.session_state.get('pred_cache', {}).get(key)
    if item and item.get('hash') == df_hash:
        return item.get('preds')
    return None
