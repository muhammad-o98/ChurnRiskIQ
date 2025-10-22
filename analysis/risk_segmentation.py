"""Risk segmentation based on churn probabilities and SHAP explanations."""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SEGMENTS = [
    ('Very High Risk', 0.9, 1.0),
    ('High Risk', 0.7, 0.9),
    ('Medium Risk', 0.4, 0.7),
    ('Low Risk', 0.0, 0.4),
]


def segment_by_probability(probs: np.ndarray) -> np.ndarray:
    # Use percentiles defined in prompt: top 10% -> Very High
    p = np.array(probs)
    q90 = np.quantile(p, 0.9)
    q70 = np.quantile(p, 0.7)
    q40 = np.quantile(p, 0.4)
    seg = []
    for v in p:
        if v >= q90:
            seg.append('Very High Risk')
        elif v >= q70:
            seg.append('High Risk')
        elif v >= q40:
            seg.append('Medium Risk')
        else:
            seg.append('Low Risk')
    return np.array(seg)


def summarize_segments(df: pd.DataFrame, prob_col: str = 'churn_probability', actual_col: str = None) -> pd.DataFrame:
    g = df.groupby('risk_segment').agg(
        customer_count=(prob_col, 'count'),
        avg_churn_probability=(prob_col, 'mean'),
    ).reset_index()
    if actual_col and actual_col in df.columns:
        g['actual_churned'] = df.groupby('risk_segment')[actual_col].sum().values
        g['churn_rate'] = df.groupby('risk_segment')[actual_col].mean().values
    g['percentage'] = (g['customer_count'] / len(df) * 100).round(2)
    return g.sort_values('avg_churn_probability', ascending=False)


def recommended_interventions(risk_level: str) -> Dict[str, str]:
    strategies = {
        'Very High Risk': {
            'discount': '30% retention offer',
            'contact': 'Immediate manager call',
            'priority': 'Critical',
        },
        'High Risk': {
            'discount': '20% loyalty discount',
            'contact': 'Personalized email campaign',
            'priority': 'High',
        },
        'Medium Risk': {
            'discount': '10% upgrade offer',
            'contact': 'Automated engagement',
            'priority': 'Normal',
        },
        'Low Risk': {
            'discount': 'No discount',
            'contact': 'Periodic newsletter',
            'priority': 'Low',
        },
    }
    return strategies.get(risk_level, {})


def top_k_risk_factors(shap_vals: np.ndarray, feature_names: List[str], k: int = 3) -> List[str]:
    idx = np.argsort(np.abs(shap_vals))[::-1][:k]
    return [feature_names[i] for i in idx]
