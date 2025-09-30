import os
import argparse
import json
import logging
import time
from typing import Dict, Any, Optional, List

import yaml
import pandas as pd

# sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# optional learners
try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover
    XGBClassifier = None
try:
    from lightgbm import LGBMClassifier
except Exception:  # pragma: no cover
    LGBMClassifier = None
try:
    from catboost import CatBoostClassifier
except Exception:  # pragma: no cover
    CatBoostClassifier = None

from data_preprocessing import prepare_data, setup_logger as setup_pp_logger
from model_training import train_with_cv, save_model_artifacts, build_pipeline, setup_logger as setup_train_logger
from model_evaluation import evaluate_classifier, setup_logger as setup_eval_logger
from hyperparameter_tuning import run_grid_search, run_random_search, setup_logger as setup_tune_logger
from model_ensembling import (
    evaluate_soft_voting,
    setup_logger as setup_ens_logger,
    SoftVotingEnsemblePredictor,
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


def build_estimator(model_key: str, cfg: Dict[str, Any]):
    params = cfg.get("params", {}) or {}
    registry = {
        # Decision Trees
        "decision_tree_gini": (DecisionTreeClassifier, {}),
        "decision_tree_entropy": (DecisionTreeClassifier, {}),
        # Random Forest
        "rf_basic": (RandomForestClassifier, {}),
        "rf_balanced": (RandomForestClassifier, {}),
        # Gradient Boosting
        "gb_basic": (GradientBoostingClassifier, {}),
        "gb_tuned": (GradientBoostingClassifier, {}),
        # XGBoost
        "xgb_basic": (XGBClassifier, {}),
        "xgb_tuned": (XGBClassifier, {}),
        # LightGBM
        "lgbm_basic": (LGBMClassifier, {}),
        "lgbm_tuned": (LGBMClassifier, {}),
        # CatBoost
        "cat_basic": (CatBoostClassifier, {}),
        "cat_tuned": (CatBoostClassifier, {}),
        # SVC
        "svc": (SVC, {}),
    }
    if model_key not in registry:
        raise ValueError(f"Unknown model key: {model_key}. Available: {list(registry.keys())}")
    Estimator, extras = registry[model_key]
    if Estimator is None:
        raise ImportError(f"Requested model '{model_key}' requires an optional dependency (check requirements).")
    # CatBoost silent default
    if Estimator is CatBoostClassifier and "verbose" not in params:
        params["verbose"] = 0
        params.setdefault("random_seed", 42)
    return Estimator(**{**extras, **params})


def select_best_model(metrics_rows: List[Dict[str, Any]], metric: str = "roc_auc") -> Dict[str, Any]:
    valid = [r for r in metrics_rows if metric in r]
    if not valid:
        raise ValueError(f"No metric '{metric}' found in any rows.")
    return max(valid, key=lambda r: r[metric])


def main(
    config_path: str,
    mode: str,
    metric: str,
    only_models: Optional[List[str]] = None,
    tune_strategy: str = "grid",
    random_iter: int = 50,
):
    logger = setup_logger("pipeline")
    logger.info(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    project = cfg.get("project", {})
    rng = project.get("random_state", 42)
    output_dir = project.get("output_dir", "artifacts")

    data_dir = os.path.join(output_dir, "data", "processed")
    models_root = os.path.join(output_dir, "models")
    reports_dir = os.path.join(output_dir, "reports")  # summary-level
    ensure_dir(data_dir); ensure_dir(models_root); ensure_dir(reports_dir)

    data_cfg = cfg.get("data", {})
    data_path = data_cfg["path"]
    target = data_cfg["target"]
    test_size = data_cfg.get("test_size", 0.2)
    stratify = data_cfg.get("stratify", True)
    pos_label = data_cfg.get("positive_label")

    preprocessing_cfg = cfg.get("preprocessing", {})

    logger.info("Preparing data...")
    full_df = pd.read_csv(data_path)
    y_full_raw = full_df[target]
    y_full = y_full_raw.map({"Yes": 1, "No": 0}).astype("Int64").fillna(0).astype(int) if y_full_raw.dtype == object else y_full_raw
    X_full = full_df.drop(columns=[target])

    preprocessor, X_train_trans, y_train, X_test_trans, y_test, feature_names, base_splits = prepare_data(
        data_path=data_path,
        target=target,
        test_size=test_size,
        random_state=rng,
        stratify=stratify,
        preprocessing_cfg=preprocessing_cfg,
        processed_out_dir=data_dir,
        logger=setup_pp_logger("preprocessing"),
    )

    from sklearn.model_selection import train_test_split as sk_split
    stratify_arr = y_full if stratify else None
    X_train_raw, X_test_raw, y_train_raw, y_test_raw = sk_split(
        X_full, y_full, test_size=test_size, random_state=rng, stratify=stratify_arr
    )

    models_cfg_all = cfg.get("models", {}).get("base_models", {})
    models_cfg = {k: v for k, v in models_cfg_all.items() if (not only_models or k in only_models)}
    if not models_cfg:
        raise RuntimeError("No models selected. Check config.models.base_models or provide --models keys.")

    training_cfg = cfg.get("training", {})
    cv_cfg = training_cfg.get("cross_validation", {})
    cv = cv_cfg.get("cv", 5)
    scoring_list = cv_cfg.get("scoring", ["roc_auc", "accuracy", "f1"])
    scoring = {s: s for s in scoring_list}
    n_jobs = cv_cfg.get("n_jobs", -1)

    tuning_cfg = cfg.get("tuning", {})
    tuning_enabled = tuning_cfg.get("enabled", True)
    param_grids = tuning_cfg.get("param_grids", {})
    if tune_strategy not in ("grid", "random"):
        tune_strategy = tuning_cfg.get("strategy", "grid")
    if tune_strategy == "random":
        random_iter = tuning_cfg.get("random_n_iter", random_iter)

    trained_models: Dict[str, Any] = {}
    summary_rows: List[Dict[str, Any]] = []

    def do_train(model_key: str, model_def: Dict[str, Any]):
        estimator = build_estimator(model_key, model_def)
        model_dir = os.path.join(models_root, model_key)
        ensure_dir(model_dir)
        logger.info(f"Model directory: {model_dir}")

        if mode in ("tune", "gridsearch", "full") and tuning_enabled and (model_key in param_grids):
            base_pipeline = build_pipeline(preprocessor, estimator)
            if (tune_strategy == "random"):
                best_pipeline, best_cv_summary = run_random_search(
                    model_name=model_key,
                    pipeline=base_pipeline,
                    X_train_raw=X_train_raw,
                    y_train=y_train_raw,
                    param_distributions=param_grids[model_key],
                    cv=cv,
                    scoring="roc_auc" if "roc_auc" in scoring else list(scoring.keys())[0],
                    n_jobs=n_jobs,
                    n_iter=random_iter,
                    out_dir=model_dir,
                    logger=setup_tune_logger(f"tuning.random.{model_key}"),
                )
            else:
                best_pipeline, best_cv_summary = run_grid_search(
                    model_name=model_key,
                    pipeline=base_pipeline,
                    X_train_raw=X_train_raw,
                    y_train=y_train_raw,
                    param_grid=param_grids[model_key],
                    cv=cv,
                    scoring="roc_auc" if "roc_auc" in scoring else list(scoring.keys())[0],
                    n_jobs=n_jobs,
                    out_dir=model_dir,
                    logger=setup_tune_logger(f"tuning.grid.{model_key}"),
                )
        else:
            best_pipeline, best_cv_summary = train_with_cv(
                preprocessor=preprocessor,
                estimator=estimator,
                X_train_raw=X_train_raw,
                y_train=y_train_raw,
                scoring=scoring,
                cv=cv,
                n_jobs=n_jobs,
                logger=setup_train_logger(f"training.{model_key}"),
            )

        save_model_artifacts(
            model_name=model_key,
            pipeline=best_pipeline,
            cv_summary=best_cv_summary,
            model_out_dir=model_dir,
            logger=setup_train_logger(f"save.{model_key}"),
        )
        return best_pipeline, model_dir

    # Train
    if mode in ("tune", "gridsearch", "train", "full"):
        for model_key, model_def in models_cfg.items():
            logger.info(f"==== Training model: {model_key} ====")
            t0 = time.time()
            pipe, model_dir = do_train(model_key, model_def)
            trained_models[model_key] = pipe
            elapsed = time.time() - t0
            with open(os.path.join(model_dir, "timing.json"), "w") as f:
                json.dump({"train_total_time_sec": elapsed}, f, indent=2)
            logger.info(f"[{model_key}] Training time: {elapsed:.2f}s")

    # Load saved for eval/pick
    if mode in ("eval", "pick-best") and not trained_models:
        import glob, joblib
        for path in glob.glob(os.path.join(models_root, "*", "model.joblib")):
            name = os.path.basename(os.path.dirname(path))
            trained_models[name] = joblib.load(path)
        if not trained_models:
            raise RuntimeError("No saved models found to evaluate/pick. Run with mode=train/full first.")

    # Evaluate
    if mode in ("eval", "full", "tune", "train", "gridsearch"):
        for model_key, model in trained_models.items():
            logger.info(f"==== Evaluating model: {model_key} ====")
            t0 = time.time()
            model_dir = os.path.join(models_root, model_key)
            metrics = evaluate_classifier(
                model_name=model_key,
                model=model,
                X_test=X_test_raw,
                y_test=y_test_raw,
                out_dir=model_dir,
                pos_label=pos_label,
                logger=setup_eval_logger(f"eval.{model_key}"),
            )
            elapsed = time.time() - t0
            with open(os.path.join(model_dir, "timing.json"), "r+") as f:
                d = json.load(f)
                d["eval_total_time_sec"] = elapsed
                f.seek(0); json.dump(d, f, indent=2); f.truncate()
            logger.info(f"[{model_key}] Evaluation time: {elapsed:.2f}s")

            row = {"model": model_key}
            row.update(metrics)
            summary_rows.append(row)

        # Optional ensemble
        ensemble_cfg = cfg.get("ensemble", {})
        if ensemble_cfg.get("enabled", True) and len(trained_models) >= 2:
            logger.info("==== Evaluating soft voting ensemble ====")
            ens_dir = os.path.join(models_root, "soft_voting")
            ensure_dir(ens_dir)
            ens_metrics = evaluate_soft_voting(
                models=trained_models,
                X_test=X_test_raw,
                y_test=y_test_raw,
                out_dir=ens_dir,
                ensemble_name="soft_voting",
                pos_label=pos_label,
                logger=setup_ens_logger("ensemble"),
            )
            # Save a savable ensemble
            model_paths = {k: os.path.join(models_root, k, "model.joblib") for k in trained_models.keys()}
            from joblib import dump
            dump(SoftVotingEnsemblePredictor(model_paths=model_paths), os.path.join(ens_dir, "model.joblib"))

            ens_row = {"model": "soft_voting"}; ens_row.update(ens_metrics)
            summary_rows.append(ens_row)

        summary_path = os.path.join(reports_dir, "summary_metrics.json")
        with open(summary_path, "w") as f:
            json.dump(summary_rows, f, indent=2)
        logger.info(f"Wrote summary metrics to {summary_path}")

    # Pick best
    if mode in ("pick-best", "full", "eval"):
        if not summary_rows:
            summary_path = os.path.join(reports_dir, "summary_metrics.json")
            if not os.path.exists(summary_path):
                raise RuntimeError("No summary_metrics.json found to pick best from.")
            with open(summary_path, "r") as f:
                summary_rows = json.load(f)

        best = select_best_model(summary_rows, metric=metric)
        best_name = best["model"]
        logger.info(f"Best model by {metric}: {best_name} with {metric}={best.get(metric)}")

        import shutil
        src_path = os.path.join(models_root, best_name, "model.joblib")
        best_dir = os.path.join(models_root, "best")
        ensure_dir(best_dir)
        dst_path = os.path.join(best_dir, "best_model.joblib")
        if os.path.exists(src_path):
            shutil.copy2(src_path, dst_path)
            with open(os.path.join(best_dir, "best_model_name.txt"), "w") as f:
                f.write(best_name)
            logger.info(f"Saved best model copy to {dst_path}")
        else:
            logger.warning(f"Model file not found for {best_name} at {src_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telecom Churn: Modular ML Pipeline")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "tune", "gridsearch", "train", "eval", "pick-best"],
        help="Pipeline mode",
    )
    parser.add_argument("--metric", type=str, default="roc_auc", help="Metric for best-model selection")
    parser.add_argument("--models", type=str, nargs="*", default=None, help="Limit to specific model keys")
    parser.add_argument("--tune-strategy", type=str, default="grid", choices=["grid", "random"], help="Tuning strategy")
    parser.add_argument("--random-iter", type=int, default=50, help="n_iter for randomized search (if used)")
    args = parser.parse_args()
    main(args.config, args.mode, args.metric, args.models, args.tune_strategy, args.random_iter)