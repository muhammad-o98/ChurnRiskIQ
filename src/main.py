#!/usr/bin/env python3
"""
Main orchestrator for the Telco Churn ML project.

Modes:
- quick: run ALL configured models without tuning, then pick best
- quick-grid: run ALL configured models with GridSearchCV, then pick best
- full: end-to-end honoring config (including tuning strategy)
- eval-pick: evaluate all saved models, then pick best
- custom: user-specified stages and/or model subset
"""
import os
import sys
import json
import time
import argparse
from typing import List, Optional

from tqdm import tqdm

# Ensure we can import modules from the src folder when running "python src/main.py"
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# Import churn_pipeline in a way that works for both run styles
try:
    import churn_pipeline as cp
except Exception as e1:
    try:
        import src.churn_pipeline as cp
    except Exception as e2:
        print("Error importing churn_pipeline:")
        print(f" - First attempt (churn_pipeline): {repr(e1)}")
        print(f" - Fallback (src.churn_pipeline): {repr(e2)}")
        print("Make sure you run from the repo root. Try: python -m src.main --mode quick")
        sys.exit(1)


def _all_model_keys(config_path: str) -> List[str]:
    import yaml
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return list((cfg.get("models", {}) or {}).get("base_models", {}).keys())


def run_quick(config_path: str, metric: str):
    # All models, no tuning. Train already evaluates; avoid duplicate eval.
    models = _all_model_keys(config_path)
    cp.main(config_path, "train", metric, models, tune_strategy="grid", random_iter=50)
    cp.main(config_path, "pick-best", metric, models, tune_strategy="grid", random_iter=50)


def run_quick_grid(config_path: str, metric: str):
    # All models with grid search; then pick best
    models = _all_model_keys(config_path)
    cp.main(config_path, "tune", metric, models, tune_strategy="grid", random_iter=50)
    cp.main(config_path, "pick-best", metric, models, tune_strategy="grid", random_iter=50)


def run_full(config_path: str, metric: str, tune_strategy: str, random_iter: int):
    cp.main(config_path, "full", metric, None, tune_strategy, random_iter)


def run_eval_pick(config_path: str, metric: str):
    models = _all_model_keys(config_path)
    cp.main(config_path, "eval", metric, models, tune_strategy="grid", random_iter=50)
    cp.main(config_path, "pick-best", metric, models, tune_strategy="grid", random_iter=50)


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
        cp.main(config_path, "pick-best", metric, models, tune_strategy, random_iter)


def predict_with_best(config_path: str, data_csv: str, model_name: Optional[str] = None, threshold: float = 0.5):
    import joblib
    import numpy as np
    import pandas as pd
    import yaml

    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    artifacts = (cfg.get("project", {}) or {}).get("output_dir", "artifacts")
    models_root = os.path.join(artifacts, "models")

    if model_name:
        model_path = os.path.join(models_root, model_name, "model.joblib")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_name} not found at {model_path}")
    else:
        model_path = os.path.join(models_root, "best", "best_model.joblib")
        if not os.path.exists(model_path):
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

    out_dir = os.path.join(artifacts, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, f"preds_{int(time.time())}.csv")
    import pandas as pd
    out_df = pd.DataFrame({"prediction": y_pred})
    if y_proba is not None:
        out_df["probability"] = y_proba
    out_df.to_csv(out_csv, index=False)
    print(f"Saved predictions to {out_csv}")


def main():
    parser = argparse.ArgumentParser(description="Telco Churn – Main Orchestrator")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config YAML")
    parser.add_argument("--mode", type=str, choices=["quick", "quick-grid", "full", "eval-pick", "custom"], default="quick", help="Execution mode")
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
        run_quick(args.config, args.metric)
    elif args.mode == "quick-grid":
        run_quick_grid(args.config, args.metric)
    elif args.mode == "full":
        run_full(args.config, args.metric, args.tune_strategy, args.random_iter)
    elif args.mode == "eval-pick":
        run_eval_pick(args.config, args.metric)
    elif args.mode == "custom":
        stages = [s.strip() for s in (args.stages or "").split(",") if s.strip()] if args.stages else None
        run_custom(args.config, args.models, args.metric, stages, args.tune_strategy, args.random_iter)

    elapsed = time.time() - t0
    print(f"Pipeline finished in {elapsed:.2f}s")


if __name__ == "__main__":
    main()