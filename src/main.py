#!/usr/bin/env python3
"""
Main orchestrator for the Telco Churn ML project.

Features:
- Modes: quick, full, custom
- Stages: prep, tune, train, eval, pick-best
- Model selection: use --models to target specific keys
- Tuning strategy: grid or random
- Prediction on new CSVs with saved best model or a chosen model
- Progress bar and timing logs

This orchestrator delegates work to churn_pipeline.py to keep logic centralized.
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from tqdm import tqdm

# Import churn_pipeline with a robust fallback
try:
    import churn_pipeline as cp  # if running "python src/main.py" from repo root
except Exception:
    try:
        import src.churn_pipeline as cp  # if running "python -m src.main"
    except Exception as e:
        print(f"Error importing churn_pipeline: {e}")
        print("Make sure you're running from the project root and 'src' is on PYTHONPATH.")
        sys.exit(1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def run_stages(
    config_path: str,
    stages: List[str],
    models: Optional[List[str]],
    metric: str,
    tune_strategy: str,
    random_iter: int,
):
    ordered = []
    for s in stages:
        s = s.lower().strip()
        if s in ("prep", "prepare", "preprocess"):
            ordered.append(("noop", {}))
        elif s in ("tune", "gridsearch"):
            ordered.append(("tune", {}))
        elif s == "train":
            ordered.append(("train", {}))
        elif s in ("eval", "evaluate"):
            ordered.append(("eval", {}))
        elif s in ("pick", "pick-best", "best"):
            ordered.append(("pick-best", {}))
        elif s == "full":
            ordered.append(("full", {}))
        else:
            raise ValueError(f"Unknown stage: {s}")

    for mode, _ in tqdm(ordered, desc="Stages", unit="stage"):
        if mode == "noop":
            continue
        cp.main(
            config_path=config_path,
            mode=mode,
            metric=metric,
            only_models=models,
            tune_strategy=tune_strategy,
            random_iter=random_iter,
        )


def run_quick(config_path: str, metric: str, tune_strategy: str, random_iter: int):
    quick_models = ["rf_basic", "gb_basic", "xgb_basic", "lgbm_basic", "svc"]
    cp.main(config_path, "train", metric, quick_models, tune_strategy, random_iter)
    cp.main(config_path, "eval", metric, quick_models, tune_strategy, random_iter)
    cp.main(config_path, "pick-best", metric, quick_models, tune_strategy, random_iter)


def run_full(config_path: str, metric: str, tune_strategy: str, random_iter: int):
    cp.main(config_path, "full", metric, None, tune_strategy, random_iter)


def run_custom(
    config_path: str,
    models: Optional[List[str]],
    metric: str,
    stages: Optional[List[str]],
    tune_strategy: str,
    random_iter: int,
):
    if stages:
        run_stages(config_path, stages, models, metric, tune_strategy, random_iter)
    else:
        cp.main(config_path, "train", metric, models, tune_strategy, random_iter)
        cp.main(config_path, "eval", metric, models, tune_strategy, random_iter)
        cp.main(config_path, "pick-best", metric, models, tune_strategy, random_iter)


def predict_with_best(config_path: str, data_csv: str, out_dir: Optional[str] = None, model_name: Optional[str] = None, threshold: float = 0.5):
    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    project = cfg.get("project", {})
    artifacts = Path(project.get("output_dir", "artifacts"))
    models_root = artifacts / "models"

    if model_name:
        model_path = models_root / model_name / "model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    else:
        model_path = models_root / "best" / "best_model.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"No best model found at {model_path}. Run pick-best first or specify --model.")

    pipe = joblib.load(model_path)
    df = pd.read_csv(data_csv)

    y_pred = pipe.predict(df)
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(df)[:, 1]
    elif hasattr(pipe, "decision_function"):
        z = pipe.decision_function(df)
        z_min, z_max = float(np.min(z)), float(np.max(z))
        y_proba = (z - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z, dtype=float)
    else:
        y_proba = None

    if y_proba is not None:
        y_pred = (y_proba >= float(threshold)).astype(int)

    out_dir = Path(out_dir) if out_dir else (artifacts / "predictions")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"preds_{timestamp()}.csv"

    out_df = pd.DataFrame({"prediction": y_pred})
    if y_proba is not None:
        out_df["probability"] = y_proba
    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Telco Churn – Main Orchestrator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["quick", "full", "custom"], default="quick", help="Execution mode")
    parser.add_argument("--stages", type=str, help="Comma-separated stages for custom mode: prep,tune,train,eval,pick")
    parser.add_argument("--models", nargs="*", default=None, help="Limit to specific model keys (custom mode)")
    parser.add_argument("--metric", type=str, default="roc_auc", help="Metric used for best-model selection")
    parser.add_argument("--tune-strategy", type=str, choices=["grid", "random"], default="grid", help="Tuning strategy")
    parser.add_argument("--random-iter", type=int, default=50, help="n_iter for randomized search")
    parser.add_argument("--predict", type=str, default=None, help="CSV path to run predictions on")
    parser.add_argument("--model", type=str, default=None, help="Specific model key to use for prediction")
    parser.add_argument("--threshold", type=float, default=0.5, help="Decision threshold for probability outputs")

    args = parser.parse_args()

    t0 = time.time()

    if args.predict:
        predict_with_best(args.config, args.predict, model_name=args.model, threshold=args.threshold)
        return

    if args.mode == "quick":
        run_quick(args.config, args.metric, args.tune_strategy, args.random_iter)
    elif args.mode == "full":
        run_full(args.config, args.metric, args.tune_strategy, args.random_iter)
    elif args.mode == "custom":
        stages = [s.strip() for s in (args.stages or "").split(",") if s.strip()] if args.stages else None
        run_custom(args.config, args.models, args.metric, stages, args.tune_strategy, args.random_iter)

    elapsed = time.time() - t0
    print(f"Pipeline finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()