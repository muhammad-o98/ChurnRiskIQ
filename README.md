# Telecom Churn – Full-Scale Config-Driven ML Pipeline

This project delivers a production-style, config-driven ML pipeline for Telco churn with:
- Your complete model suite: Decision Trees (gini/entropy), Random Forest (basic/balanced), Gradient Boosting (basic/tuned), XGBoost (basic/tuned), LightGBM (basic/tuned), CatBoost (basic/tuned), plus SVC
- Feature engineering aligned to the data (tenure_group, LongTerm, service normalization, AvgCharge, Yeo–Johnson)
- Data preprocessing with OneHot(drop='first') for categoricals and passthrough numerics
- Training with cross-validation, per-model artifact folders and figures
- Hyperparameter tuning (grid or randomized)
- Extended evaluation with threshold analysis and plots
- Soft-voting ensemble (optional)
- Best model selection and copy
- A main orchestrator for quick/full/custom runs and prediction

## Install
```bash
pip install -r requirements.txt
```

## Configure
Edit [config/config.yaml](config/config.yaml):
- data.path: path to WA_Fn-UseC_-Telco-Customer-Churn.csv
- models.base_models: choose which models to run (defaults mirror your clf1–clf12 + svc)
- tuning.param_grids: ready-made grids derived from your notebook
- tuning.strategy: "grid" or "random"
- preprocessing: feature engineering and encoding options

## Orchestrator (recommended)
Use the new main orchestrator for high-level control.

Quick mode (fast subset, no tuning):
```bash
python src/main.py --mode quick
```

Full mode (full pipeline, including tuning if enabled in config):
```bash
python src/main.py --mode full
```

Custom mode (select models and/or stages):
```bash
# Train -> Eval -> Pick only specific models
python src/main.py --mode custom --models xgb_tuned lgbm_tuned svc

# Explicit stages (any subset of prep,tune,train,eval,pick), in order:
python src/main.py --mode custom --stages prep,train,eval,pick
python src/main.py --mode custom --stages tune,eval --tune-strategy random --random-iter 50
```

Predict on new CSV using the best saved model (or a specific model):
```bash
# After pick-best
python src/main.py --predict data/new_customers.csv

# Use a specific model folder (artifacts/models/<model_key>/model.joblib)
python src/main.py --predict data/new_customers.csv --model xgb_tuned --threshold 0.45
```

## Core pipeline (advanced)
You can also call the core pipeline directly:
```bash
# End-to-end flow
python src/churn_pipeline.py --config config/config.yaml --mode full

# Train without tuning
python src/churn_pipeline.py --mode train

# Grid search
python src/churn_pipeline.py --mode tune --tune-strategy grid

# Randomized search
python src/churn_pipeline.py --mode tune --tune-strategy random --random-iter 50

# Evaluate saved models
python src/churn_pipeline.py --mode eval

# Pick best by metric
python src/churn_pipeline.py --mode pick-best --metric f1

# Limit to certain models in any mode
python src/churn_pipeline.py --mode full --models xgb_tuned lgbm_tuned svc
```

## Artifacts and Structure
- Per-model directory: `artifacts/models/<model_key>/`
  - `model.joblib` – fitted pipeline (preprocessing + estimator)
  - `cv_summary.json` – CV/tuning summary with timing
  - `best_params.json` + `gridsearch_results.csv` or `randomsearch_results.csv` – if tuning ran
  - `metrics.json` – test metrics including `best_f1_threshold`
  - `classification_report.txt`
  - `timing.json` – training/evaluation durations
  - `figures/`
    - `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
    - `f1_vs_threshold.png`, `cm_elements_vs_threshold.png`
    - `prob_by_actual_class.png`, `prob_by_outcome.png`
- Ensemble: `artifacts/models/soft_voting/soft_voting_metrics.json`
- Project summary: `artifacts/reports/summary_metrics.json`
- Best model copy: `artifacts/models/best/best_model.joblib`, `best_model_name.txt`
- Predictions: `artifacts/predictions/preds_<timestamp>.csv`

## Notes on performance and parallelism
- CV and many estimators already use all cores via `n_jobs=-1` where applicable.
- Training many heavy models (e.g., RF with 20k trees, GB with 20k estimators) can be time-consuming. Start with smaller configs in `config.yaml`, iterate, then scale up.
- If you need “parallel across models,” we can add a parallel launcher that runs subsets in separate processes. Since each model often parallelizes internally, nested parallelism can be counterproductive; we can tune this for your environment if needed.
