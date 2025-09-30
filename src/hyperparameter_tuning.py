import os
import json
import logging
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV


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


def prefix_param_grid(param_grid: Dict[str, Any], prefix: str = "clf__") -> Dict[str, Any]:
    return {f"{prefix}{k}": v for k, v in param_grid.items()}


def run_grid_search(
    model_name: str,
    pipeline: Pipeline,
    X_train_raw,
    y_train,
    param_grid: Dict[str, Any],
    cv: int,
    scoring: str,
    n_jobs: int,
    out_dir: str,
    logger: logging.Logger = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    ensure_dir(out_dir)
    if logger:
        logger.info(f"Grid search for {model_name} with scoring={scoring}, cv={cv}, n_jobs={n_jobs}")
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=prefix_param_grid(param_grid),
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        verbose=1,
    )
    grid.fit(X_train_raw, y_train)

    best_estimator = grid.best_estimator_
    best_summary = {
        "best_score": float(grid.best_score_),
        "best_params": grid.best_params_,
        "scoring": scoring,
        "cv": cv,
        "strategy": "grid",
    }

    with open(os.path.join(out_dir, f"best_params.json"), "w") as f:
        json.dump(best_summary, f, indent=2)
    results_df = pd.DataFrame(grid.cv_results_)
    results_df.to_csv(os.path.join(out_dir, f"gridsearch_results.csv"), index=False)

    if logger:
        logger.info(f"Best score for {model_name}: {best_summary['best_score']}")
    return best_estimator, best_summary


def run_random_search(
    model_name: str,
    pipeline: Pipeline,
    X_train_raw,
    y_train,
    param_distributions: Dict[str, Any],
    cv: int,
    scoring: str,
    n_jobs: int,
    n_iter: int,
    out_dir: str,
    logger: Optional[logging.Logger] = None,
) -> Tuple[Pipeline, Dict[str, Any]]:
    ensure_dir(out_dir)
    if logger:
        logger.info(f"Randomized search for {model_name} scoring={scoring}, cv={cv}, n_jobs={n_jobs}, n_iter={n_iter}")
    rnd = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=prefix_param_grid(param_distributions),
        n_iter=n_iter,
        scoring=scoring,
        cv=cv,
        n_jobs=n_jobs,
        refit=True,
        return_train_score=False,
        verbose=1,
        random_state=42,
    )
    rnd.fit(X_train_raw, y_train)

    best_estimator = rnd.best_estimator_
    best_summary = {
        "best_score": float(rnd.best_score_),
        "best_params": rnd.best_params_,
        "scoring": scoring,
        "cv": cv,
        "strategy": "random",
        "n_iter": n_iter,
    }

    with open(os.path.join(out_dir, f"best_params.json"), "w") as f:
        json.dump(best_summary, f, indent=2)
    results_df = pd.DataFrame(rnd.cv_results_)
    results_df.to_csv(os.path.join(out_dir, f"randomsearch_results.csv"), index=False)

    if logger:
        logger.info(f"Best score for {model_name}: {best_summary['best_score']}")
    return best_estimator, best_summary