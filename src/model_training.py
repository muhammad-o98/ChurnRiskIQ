import os
import json
import logging
import time
from typing import Dict, Any, Tuple

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_validate


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


def build_pipeline(preprocessor: Pipeline, estimator: BaseEstimator) -> Pipeline:
    pipe = Pipeline(steps=[("preprocessor", preprocessor), ("clf", estimator)])
    return pipe


def train_with_cv(
    preprocessor: Pipeline,
    estimator: BaseEstimator,
    X_train_raw,
    y_train,
    scoring: Dict[str, str],
    cv: int = 5,
    n_jobs: int = -1,
    logger: logging.Logger = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    pipe = build_pipeline(preprocessor, estimator)
    if logger:
        logger.info(f"Cross-validating with scoring={list(scoring.keys())}, cv={cv}, n_jobs={n_jobs}")
    t0 = time.time()
    cv_results = cross_validate(
        pipe, X_train_raw, y_train, scoring=scoring, cv=cv, n_jobs=n_jobs, return_train_score=False
    )
    cv_time = time.time() - t0
    summary = {metric: float(np.mean(scores)) for metric, scores in cv_results.items() if metric.startswith("test_")}
    summary["cv_time_sec"] = cv_time
    if logger:
        logger.info(f"CV summary: {summary}")

    t1 = time.time()
    pipe.fit(X_train_raw, y_train)
    fit_time = time.time() - t1
    summary["fit_time_sec"] = fit_time
    return pipe, summary


def save_model_artifacts(
    model_name: str,
    pipeline: Pipeline,
    cv_summary: Dict[str, Any],
    model_out_dir: str,
    joblib_module=None,
    logger: logging.Logger = None,
) -> None:
    ensure_dir(model_out_dir)
    if logger:
        logger.info(f"Saving model artifacts to {model_out_dir}")
    if joblib_module is None:
        import joblib as joblib_module  # type: ignore

    model_path = os.path.join(model_out_dir, "model.joblib")
    joblib_module.dump(pipeline, model_path)
    with open(os.path.join(model_out_dir, "cv_summary.json"), "w") as f:
        json.dump(cv_summary, f, indent=2)