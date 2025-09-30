# Telecom Churn – Full-Scale Config-Driven ML Pipeline

Production-style, config-driven pipeline for Telco churn:
- Models: Decision Trees (gini/entropy), Random Forest (basic/balanced), Gradient Boosting (basic/tuned), XGBoost (basic/tuned), LightGBM (basic/tuned), CatBoost (basic/tuned), plus SVC
- Feature engineering aligned to your notebook (tenure_group, LongTerm, service normalization, AvgCharge, optional Yeo–Johnson)
- Cross-validated training, grid/randomized hyperparameter tuning
- Per-model metrics and plots, soft-voting ensemble saved as a model
- Best-model selection and copy to artifacts/models/best/
- Explainability: tree visualization, SHAP (tree models), threshold insights, probability diagnostics

## Install
```bash
pip install -r requirements.txt
```

## Configure
Edit [config/config.yaml](config/config.yaml):
- data.path: path to your CSV
- models.base_models: select the models to run (defaults mirror your clf1–clf12 + svc)
- tuning.param_grids: ready-made grids
- tuning.strategy: "grid" or "random"
- preprocessing: feature engineering and encoding options

## Orchestrator – run modes (src/main.py)
- Quick: run ALL configured models (no tuning), then pick best
```bash
python src/main.py --mode quick
# -> train (all models, no tuning) -> pick-best
```

- Quick Grid: run ALL models with GridSearchCV, then pick best
```bash
python src/main.py --mode quick-grid
# -> tune (grid) (all models) -> pick-best
```

- Full: complete pipeline honoring tuning strategy in config (grid or random)
```bash
python src/main.py --mode full
# -> prepare → tune/train → evaluate (+plots) → ensemble → pick best
```

- Evaluate and pick (use already-saved models)
```bash
python src/main.py --mode eval-pick
# -> eval all saved models -> pick best
```

- Custom (select stages and/or model subset)
```bash
# By stages (any subset of prep,tune,train,eval,pick), in order:
python src/main.py --mode custom --stages train,eval,pick

# By model keys:
python src/main.py --mode custom --models xgb_tuned lgbm_tuned svc
```

Prediction on new CSV (uses best model unless --model is specified):
```bash
python src/main.py --predict data/new_customers.csv
python src/main.py --predict data/new_customers.csv --model xgb_tuned --threshold 0.45
```

## Direct pipeline (advanced) – src/churn_pipeline.py
```bash
python src/churn_pipeline.py --mode full
python src/churn_pipeline.py --mode train
python src/churn_pipeline.py --mode tune --tune-strategy grid
python src/churn_pipeline.py --mode tune --tune-strategy random --random-iter 50
python src/churn_pipeline.py --mode eval
python src/churn_pipeline.py --mode pick-best --metric f1
python src/churn_pipeline.py --mode full --models xgb_tuned lgbm_tuned svc
```

## Artifacts and Outputs
- Per-model directory: artifacts/models/<model_key>/
  - model.joblib – fitted pipeline
  - cv_summary.json – CV/tuning summary with timing
  - best_params.json + gridsearch_results.csv or randomsearch_results.csv (if tuned)
  - metrics.json – test metrics including best_f1_threshold
  - prob_threshold_insights.json + prob_threshold_insights.txt
  - classification_report.txt
  - timing.json – training/evaluation durations
  - figures/
    - confusion_matrix.png, roc_curve.png, pr_curve.png
    - f1_vs_threshold.png, cm_elements_vs_threshold.png
    - prob_by_actual_class.png, prob_by_outcome.png, prob_boxplot_by_outcome.png
    - feature_importances_top20.png (if available)
    - tree_top.png / rf_best_tree_top.png / gb_first_tree_top.png (where applicable)
    - shap_beeswarm.png, shap_bar.png, shap_dependence_*.png (for tree-based models)
  - shap_top_features.json (top features by |SHAP|)

- Ensemble: artifacts/models/soft_voting/model.joblib and soft_voting_metrics.json  
- Summary: artifacts/reports/summary_metrics.json  
- Best model: artifacts/models/best/best_model.joblib, best_model_name.txt  
- Predictions: artifacts/predictions/preds_<timestamp>.csv

## Notes
- SHAP is computed on a sample (max 500) for performance and only for tree-based models by default.
- Tree visualizations plot top levels for readability; adjust depth in code if needed.
- Some models parallelize internally; avoid heavy nested parallelism.
