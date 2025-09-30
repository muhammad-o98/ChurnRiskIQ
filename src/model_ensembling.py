import os
import json
import logging
from typing import Dict, Any

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s"))
        logger.addHandler(ch)
    return logger


def soft_voting_predict_proba(models: Dict[str, Any], X) -> np.ndarray:
    probas = []
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            probas.append(model.predict_proba(X)[:, 1])
        elif hasattr(model, "decision_function"):
            z = model.decision_function(X)
            z_min, z_max = np.min(z), np.max(z)
            p = (z - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z)
            probas.append(p)
        else:
            probas.append(model.predict(X).astype(float))
    return np.mean(np.column_stack(probas), axis=1)


def evaluate_soft_voting(
    models: Dict[str, Any],
    X_test,
    y_test,
    out_dir: str,
    ensemble_name: str = "soft_voting",
    pos_label=None,
    logger: logging.Logger = None,
) -> Dict[str, float]:
    ensure_dir(out_dir)
    y_proba = soft_voting_predict_proba(models, X_test)
    y_pred = (y_proba >= 0.5).astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
    except Exception:
        pass

    with open(os.path.join(out_dir, f"{ensemble_name}_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    if logger:
        logger.info(f"[{ensemble_name}] Test metrics: {metrics}")
    return metrics