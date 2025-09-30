import os
import json
import logging
from typing import Dict, Any, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    classification_report,
)


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


def get_probabilities(model, X) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        z = model.decision_function(X)
        z_min, z_max = np.min(z), np.max(z)
        if z_max > z_min:
            return (z - z_min) / (z_max - z_min)
    return None


def _plot_and_save(fig_path: str):
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()


def evaluate_classifier_basic(
    model_name: str,
    model,
    X_test,
    y_test,
    out_dir: str,
    pos_label=None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    y_pred = model.predict(X_test)
    y_proba = get_probabilities(model, X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
    }
    if y_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass

    if logger:
        logger.info(f"[{model_name}] Test metrics: {metrics}")

    with open(os.path.join(out_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, f"classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _plot_and_save(os.path.join(fig_dir, f"confusion_matrix.png"))

    if y_proba is not None and len(np.unique(y_test)) == 2:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{model_name} - ROC Curve")
        _plot_and_save(os.path.join(fig_dir, f"roc_curve.png"))

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{model_name} - Precision-Recall Curve")
        _plot_and_save(os.path.join(fig_dir, f"pr_curve.png"))

    return metrics


def evaluate_classifier_full(
    model_name: str,
    model,
    X_test,
    y_test,
    out_dir: str,
    pos_label=None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Full evaluation:
    - Basic metrics + CM, ROC, PR
    - Best F1 threshold search and plots:
      * F1 vs Threshold
      * TP/FN/FP/TN vs Threshold
      * Probability distributions by actual class
      * Probability distributions by outcome groups (TP, TN, FP, FN)
    """
    metrics = evaluate_classifier_basic(
        model_name=model_name,
        model=model,
        X_test=X_test,
        y_test=y_test,
        out_dir=out_dir,
        pos_label=pos_label,
        logger=logger,
    )

    y_proba = get_probabilities(model, X_test)
    if y_proba is None:
        return metrics

    fig_dir = os.path.join(out_dir, "figures")
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))
    best_f1_thresh = float(thresholds[int(np.argmax(f1_scores))])
    best_f1_score = float(np.max(f1_scores))
    metrics["best_f1_threshold"] = best_f1_thresh
    metrics["best_f1_score"] = best_f1_score

    # F1 vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} - F1 Score vs Threshold")
    plt.grid(True, alpha=0.3)
    _plot_and_save(os.path.join(fig_dir, "f1_vs_threshold.png"))

    # CM elements vs Threshold
    tp_list, tn_list, fp_list, fn_list = [], [], [], []
    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        tn_list.append(cm[0, 0])
        fp_list.append(cm[0, 1])
        fn_list.append(cm[1, 0])
        tp_list.append(cm[1, 1])

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, tp_list, label="True Positives", color="green")
    plt.plot(thresholds, tn_list, label="True Negatives", color="blue")
    plt.plot(thresholds, fp_list, label="False Positives", color="orange")
    plt.plot(thresholds, fn_list, label="False Negatives", color="red")
    plt.axvline(best_f1_thresh, color="purple", linestyle="--", label=f"Best F1 Thresh {best_f1_thresh:.3f}")
    plt.xlabel("Threshold")
    plt.ylabel("Count")
    plt.title(f"{model_name} - Confusion Matrix Elements vs Threshold")
    plt.legend(loc="best")
    plt.grid(True, alpha=0.3)
    _plot_and_save(os.path.join(fig_dir, "cm_elements_vs_threshold.png"))

    # Probability distribution by actual class
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba[y_test == 1], bins=50, alpha=0.5, label="Actual Churn", color="red", density=True)
    plt.hist(y_proba[y_test == 0], bins=50, alpha=0.5, label="No Churn", color="blue", density=True)
    plt.axvline(best_f1_thresh, color="green", linestyle="--", linewidth=2, label=f"Best F1 Thresh ({best_f1_thresh:.3f})")
    plt.axvline(0.5, color="gray", linestyle=":", linewidth=1.5, label="Default 0.5")
    plt.xlabel("Predicted Probability of Churn")
    plt.ylabel("Density")
    plt.title(f"{model_name} - Probability Distribution by Actual Class")
    plt.legend()
    plt.grid(True, alpha=0.3)
    _plot_and_save(os.path.join(fig_dir, "prob_by_actual_class.png"))

    # Outcomes distributions (TP/TN/FP/FN) at best threshold
    y_pred_best = (y_proba >= best_f1_thresh).astype(int)
    tp_probs = y_proba[(y_test == 1) & (y_pred_best == 1)]
    tn_probs = y_proba[(y_test == 0) & (y_pred_best == 0)]
    fp_probs = y_proba[(y_test == 0) & (y_pred_best == 1)]
    fn_probs = y_proba[(y_test == 1) & (y_pred_best == 0)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(tp_probs, bins=30, alpha=0.7, color="green", edgecolor="black")
    axes[0, 0].axvline(best_f1_thresh, color="red", linestyle="--", label=f"Thresh={best_f1_thresh:.2f}")
    axes[0, 0].set_title(f"True Positives (n={len(tp_probs)})"); axes[0, 0].legend()

    axes[0, 1].hist(tn_probs, bins=30, alpha=0.7, color="blue", edgecolor="black")
    axes[0, 1].axvline(best_f1_thresh, color="red", linestyle="--", label=f"Thresh={best_f1_thresh:.2f}")
    axes[0, 1].set_title(f"True Negatives (n={len(tn_probs)})"); axes[0, 1].legend()

    axes[1, 0].hist(fp_probs, bins=30, alpha=0.7, color="orange", edgecolor="black")
    axes[1, 0].axvline(best_f1_thresh, color="red", linestyle="--", label=f"Thresh={best_f1_thresh:.2f}")
    axes[1, 0].set_title(f"False Positives (n={len(fp_probs)})"); axes[1, 0].legend()

    axes[1, 1].hist(fn_probs, bins=30, alpha=0.7, color="red", edgecolor="black")
    axes[1, 1].axvline(best_f1_thresh, color="red", linestyle="--", label=f"Thresh={best_f1_thresh:.2f}")
    axes[1, 1].set_title(f"False Negatives (n={len(fn_probs)})"); axes[1, 1].legend()

    for ax in axes.ravel():
        ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Count")
    fig.suptitle(f"{model_name} - Probability Distributions by Outcome", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "prob_by_outcome.png"))
    plt.close()

    with open(os.path.join(out_dir, f"metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics


# Backward-compat shim: keep old import path working
def evaluate_classifier(
    model_name: str,
    model,
    X_test,
    y_test,
    out_dir: str,
    pos_label=None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    return evaluate_classifier_full(model_name, model, X_test, y_test, out_dir, pos_label, logger)