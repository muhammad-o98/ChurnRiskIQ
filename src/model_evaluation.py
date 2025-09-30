import os
import json
import logging
from typing import Dict, Any, Optional, Tuple, List, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.tree import plot_tree, DecisionTreeClassifier
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

# Optional: SHAP for explainability
try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False


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
    plt.savefig(fig_path, bbox_inches="tight")
    plt.close()


def _get_pipeline_parts(model_or_pipe: Union[Pipeline, Any]):
    if isinstance(model_or_pipe, Pipeline):
        pre = model_or_pipe.named_steps.get("preprocessor")
        clf = model_or_pipe.named_steps.get("clf", model_or_pipe)
        return pre, clf
    return None, model_or_pipe


def _get_feature_names_from_preprocessor(preprocessor, X_sample) -> List[str]:
    # Build feature names from fitted preprocessor pipeline
    try:
        ct = preprocessor.named_steps["ct"]
        cat = ct.named_transformers_["cat"]
        enc = cat.named_steps["encoder"]
        cat_features = enc.get_feature_names_out(ct.transformers_[0][2]).tolist()  # original categorical cols
        # Numerical features are at transformers_[1][2]
        num_features = ct.transformers_[1][2]
        return list(cat_features) + list(num_features)
    except Exception:
        # Fallback to positional names
        if hasattr(X_sample, "shape"):
            return [f"f_{i}" for i in range(X_sample.shape[1])]
        return []


def _evaluate_probability_thresholds(model_name: str, y_test: np.ndarray, y_proba: np.ndarray, fig_dir: str) -> Dict[str, Any]:
    thresholds = np.linspace(0, 1, 100)
    f1_scores = []
    tp_list, tn_list, fp_list, fn_list = [], [], [], []

    for thresh in thresholds:
        y_pred_thresh = (y_proba >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred_thresh)
        tn_list.append(cm[0, 0])
        fp_list.append(cm[0, 1])
        fn_list.append(cm[1, 0])
        tp_list.append(cm[1, 1])
        f1_scores.append(f1_score(y_test, y_pred_thresh, zero_division=0))

    best_idx = int(np.argmax(f1_scores))
    best_f1_thresh = float(thresholds[best_idx])
    best_f1_score = float(f1_scores[best_idx])

    # F1 vs Threshold
    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, f1_scores, color="blue")
    plt.xlabel("Threshold")
    plt.ylabel("F1 Score")
    plt.title(f"{model_name} - F1 Score vs Threshold")
    plt.grid(True, alpha=0.3)
    _plot_and_save(os.path.join(fig_dir, "f1_vs_threshold.png"))

    # CM elements vs Threshold
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

    # Probability distribution by actual class (overlapping dists)
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
    plt.savefig(os.path.join(fig_dir, "prob_by_outcome.png"), bbox_inches="tight")
    plt.close()

    # Probability boxplot by outcome
    fig, ax = plt.subplots(figsize=(10, 6))
    prob_data, labels, colors = [], [], ["green", "blue", "orange", "red"]
    for probs, label in [(tp_probs, "True\nPositives"), (tn_probs, "True\nNegatives"), (fp_probs, "False\nPositives"), (fn_probs, "False\nNegatives")]:
        if len(probs) > 0:
            prob_data.append(probs)
            labels.append(f"{label}\n(n={len(probs)})")
    if prob_data:
        bp = ax.boxplot(prob_data, labels=labels, patch_artist=True, showmeans=True)
        for patch, color in zip(bp["boxes"], colors[:len(prob_data)]):
            patch.set_facecolor(color); patch.set_alpha(0.5)
    ax.axhline(best_f1_thresh, color="purple", linestyle="--", label=f"Best F1 Threshold ({best_f1_thresh:.3f})")
    ax.set_ylabel("Predicted Probability")
    ax.set_title("Probability Distributions by Classification Outcome")
    ax.legend()
    ax.grid(True, alpha=0.3)
    _plot_and_save(os.path.join(fig_dir, "prob_boxplot_by_outcome.png"))

    # Textual stats and overlap
    cm_best = confusion_matrix(y_test, y_pred_best)
    total = int(np.sum(cm_best))
    precision = float(cm_best[1, 1] / (cm_best[1, 1] + cm_best[0, 1])) if (cm_best[1, 1] + cm_best[0, 1]) > 0 else 0.0
    recall = float(cm_best[1, 1] / (cm_best[1, 1] + cm_best[1, 0])) if (cm_best[1, 1] + cm_best[1, 0]) > 0 else 0.0
    accuracy = float((cm_best[1, 1] + cm_best[0, 0]) / total) if total > 0 else 0.0

    prob_stats = {
        "mean_no_churn": float(np.mean(y_proba[y_test == 0])) if np.any(y_test == 0) else None,
        "mean_churn": float(np.mean(y_proba[y_test == 1])) if np.any(y_test == 1) else None,
        "median_no_churn": float(np.median(y_proba[y_test == 0])) if np.any(y_test == 0) else None,
        "median_churn": float(np.median(y_proba[y_test == 1])) if np.any(y_test == 1) else None,
        "std_no_churn": float(np.std(y_proba[y_test == 0])) if np.any(y_test == 0) else None,
        "std_churn": float(np.std(y_proba[y_test == 1])) if np.any(y_test == 1) else None,
        "min_no_churn": float(np.min(y_proba[y_test == 0])) if np.any(y_test == 0) else None,
        "max_no_churn": float(np.max(y_proba[y_test == 0])) if np.any(y_test == 0) else None,
        "min_churn": float(np.min(y_proba[y_test == 1])) if np.any(y_test == 1) else None,
        "max_churn": float(np.max(y_proba[y_test == 1])) if np.any(y_test == 1) else None,
    }
    # Overlap percentage: fraction of non-churn probs falling within churn prob range
    if np.any(y_test == 1) and np.any(y_test == 0):
        overlap_zone = np.sum((y_proba[y_test == 0] > np.min(y_proba[y_test == 1])) & (y_proba[y_test == 0] < np.max(y_proba[y_test == 1])))
        overlap_pct = float(overlap_zone / len(y_proba[y_test == 0]) * 100.0)
    else:
        overlap_pct = None

    insights = {
        "best_f1_threshold": best_f1_thresh,
        "best_f1_score": best_f1_score,
        "cm_best": cm_best.tolist(),
        "precision_at_best_f1": precision,
        "recall_at_best_f1": recall,
        "accuracy_at_best_f1": accuracy,
        "overlap_pct_nonchurn_within_churn_range": overlap_pct,
        "probability_stats": prob_stats,
        "thresholds": thresholds.tolist(),
        "tp_list": tp_list,
        "tn_list": tn_list,
        "fp_list": fp_list,
        "fn_list": fn_list,
        "f1_scores": f1_scores,
    }
    return insights


def _visualize_trees(model_name: str, pipe: Pipeline, X_test_raw, out_dir: str, logger: Optional[logging.Logger] = None, max_depth: int = 3):
    try:
        pre, clf = _get_pipeline_parts(pipe)
        if pre is None:
            # Not a pipeline; skip
            return
        # Build feature names and transform X for SHAP feature ordering consistency
        X_proc = pre.transform(X_test_raw)
        feat_names = _get_feature_names_from_preprocessor(pre, X_proc)

        fig_dir = os.path.join(out_dir, "figures")
        ensure_dir(fig_dir)

        # DecisionTree: visualize top levels
        if isinstance(clf, DecisionTreeClassifier):
            plt.figure(figsize=(20, 10))
            plot_tree(clf, feature_names=feat_names, filled=True, max_depth=max_depth, class_names=["No Churn", "Churn"])
            plt.title(f"{model_name} - Decision Tree (Top {max_depth} levels)")
            _plot_and_save(os.path.join(fig_dir, "tree_top.png"))
            return

        # RandomForest: pick the "most important" tree (sum of abs feature_importances)
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        if isinstance(clf, RandomForestClassifier) and hasattr(clf, "estimators_") and len(clf.estimators_) > 0:
            importances_sums = [float(np.sum(np.abs(getattr(t, "feature_importances_", np.zeros(len(feat_names)))))) for t in clf.estimators_]
            best_idx = int(np.argmax(importances_sums))
            best_tree = clf.estimators_[best_idx]
            plt.figure(figsize=(20, 10))
            plot_tree(best_tree, feature_names=feat_names, filled=True, max_depth=max_depth, class_names=["No Churn", "Churn"])
            plt.title(f"{model_name} - Random Forest Best Tree #{best_idx} (Top {max_depth} levels)")
            _plot_and_save(os.path.join(fig_dir, "rf_best_tree_top.png"))

        # GradientBoosting: visualize the first tree (binary classifier has shape (n_estimators, 1))
        if isinstance(clf, GradientBoostingClassifier) and hasattr(clf, "estimators_"):
            try:
                first_tree = clf.estimators_[0][0]
                plt.figure(figsize=(20, 10))
                plot_tree(first_tree, feature_names=feat_names, filled=True, max_depth=max_depth, class_names=["No Churn", "Churn"])
                plt.title(f"{model_name} - Gradient Boosting First Tree (Top {max_depth} levels)")
                _plot_and_save(os.path.join(fig_dir, "gb_first_tree_top.png"))
            except Exception:
                pass

        # Generic feature importances bar plot if available
        if hasattr(clf, "feature_importances_"):
            importances = np.array(clf.feature_importances_, dtype=float)
            order = np.argsort(importances)[::-1][:20]
            plt.figure(figsize=(8, 8))
            plt.barh(np.array(feat_names)[order][::-1], importances[order][::-1])
            plt.title(f"{model_name} - Top Feature Importances")
            plt.gca().invert_yaxis()
            _plot_and_save(os.path.join(fig_dir, "feature_importances_top20.png"))
    except Exception as e:
        if logger:
            logger.warning(f"Tree visualization skipped due to: {e}")


def _shap_analysis(model_name: str, pipe: Pipeline, X_test_raw, out_dir: str, logger: Optional[logging.Logger] = None, max_samples: int = 500):
    if not _HAS_SHAP:
        if logger:
            logger.info("SHAP not installed; skipping SHAP analysis.")
        return

    try:
        pre, clf = _get_pipeline_parts(pipe)
        if pre is None:
            return
        X_proc = pre.transform(X_test_raw)
        feat_names = _get_feature_names_from_preprocessor(pre, X_proc)

        # Sample for performance
        n = X_proc.shape[0]
        idx = np.random.RandomState(42).choice(n, size=min(max_samples, n), replace=False)
        X_s = X_proc[idx]

        # Prefer TreeExplainer for tree models; fallback to generic Explainer if available
        tree_types = ("RandomForestClassifier", "GradientBoostingClassifier", "XGBClassifier", "LGBMClassifier", "CatBoostClassifier")
        if clf.__class__.__name__ in tree_types:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer.shap_values(X_s)
        else:
            # KernelExplainer can be very slow; skip non-tree shap by default to keep runtime reasonable
            if logger:
                logger.info(f"Skipping SHAP for non-tree model {clf.__class__.__name__} to avoid long runtimes.")
            return

        fig_dir = os.path.join(out_dir, "figures")
        ensure_dir(fig_dir)

        # Binary classification outputs: pick class 1 if list is returned
        if isinstance(shap_values, list) and len(shap_values) > 1:
            sv = shap_values[1]
        else:
            sv = shap_values

        # SHAP summary beeswarm
        plt.figure()
        shap.summary_plot(sv, X_s, feature_names=feat_names, show=False)
        _plot_and_save(os.path.join(fig_dir, "shap_beeswarm.png"))

        # SHAP bar plot
        plt.figure()
        shap.summary_plot(sv, X_s, feature_names=feat_names, plot_type="bar", show=False)
        _plot_and_save(os.path.join(fig_dir, "shap_bar.png"))

        # Top feature dependence plots (up to 3)
        try:
            mean_abs = np.mean(np.abs(sv), axis=0)
            top_idx = np.argsort(mean_abs)[::-1][:3]
            for i, fi in enumerate(top_idx):
                plt.figure()
                shap.dependence_plot(fi, sv, X_s, feature_names=feat_names, show=False)
                _plot_and_save(os.path.join(fig_dir, f"shap_dependence_{i+1}_{feat_names[fi]}.png"))
        except Exception:
            pass

        # Save top features json
        mean_abs = np.mean(np.abs(sv), axis=0)
        order = np.argsort(mean_abs)[::-1][:20]
        top_features = [{"feature": feat_names[i], "mean_abs_shap": float(mean_abs[i])} for i in order]
        with open(os.path.join(out_dir, "shap_top_features.json"), "w") as f:
            json.dump(top_features, f, indent=2)

    except Exception as e:
        if logger:
            logger.warning(f"SHAP analysis skipped due to: {e}")


def evaluate_classifier(
    model_name: str,
    model,
    X_test,
    y_test,
    out_dir: str,
    pos_label=None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Extended evaluation with:
    - Metrics (accuracy, precision, recall, f1, roc_auc)
    - Confusion matrix, ROC, PR
    - Probability/threshold insights: F1 vs threshold, CM elements vs threshold,
      overlap stats, outcomes distributions, boxplot by outcome, and a JSON insights report
    - Tree visualization (DecisionTree/RandomForest/GradientBoosting)
    - SHAP analysis (tree-based models; limited sample)
    """
    ensure_dir(out_dir)
    fig_dir = os.path.join(out_dir, "figures")
    ensure_dir(fig_dir)

    # Predictions and basic metrics
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

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(out_dir, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    _plot_and_save(os.path.join(fig_dir, "confusion_matrix.png"))

    # ROC/PR
    if y_proba is not None and len(np.unique(y_test)) == 2:
        RocCurveDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{model_name} - ROC Curve")
        _plot_and_save(os.path.join(fig_dir, "roc_curve.png"))

        PrecisionRecallDisplay.from_predictions(y_test, y_proba)
        plt.title(f"{model_name} - Precision-Recall Curve")
        _plot_and_save(os.path.join(fig_dir, "pr_curve.png"))

    # Probability/threshold insights and reports
    if y_proba is not None:
        insights = _evaluate_probability_thresholds(model_name, y_test, y_proba, fig_dir)
        metrics.update({
            "best_f1_threshold": insights["best_f1_threshold"],
            "best_f1_score": insights["best_f1_score"],
        })
        # Save full insights JSON + a concise text report
        with open(os.path.join(out_dir, "prob_threshold_insights.json"), "w") as f:
            json.dump(insights, f, indent=2)
        with open(os.path.join(out_dir, "prob_threshold_insights.txt"), "w") as f:
            f.write(
                f"Best F1 threshold: {insights['best_f1_threshold']:.3f}\n"
                f"Best F1 score: {insights['best_f1_score']:.3f}\n"
                f"Precision@bestF1: {insights['precision_at_best_f1']:.3f}\n"
                f"Recall@bestF1: {insights['recall_at_best_f1']:.3f}\n"
                f"Accuracy@bestF1: {insights['accuracy_at_best_f1']:.3f}\n"
                f"Overlap % (non-churn in churn range): "
                f"{insights['overlap_pct_nonchurn_within_churn_range'] if insights['overlap_pct_nonchurn_within_churn_range'] is not None else 'NA'}\n"
            )

    # Visualize trees (if supported) and feature importances
    _visualize_trees(model_name, model, X_test, out_dir, logger=logger)

    # SHAP (tree models; sampled)
    _shap_analysis(model_name, model, X_test, out_dir, logger=logger)

    # Update metrics with any computed fields
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return metrics