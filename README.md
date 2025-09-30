# Telecom Churn – Modular, Scalable ML Project

This packages your notebook logic into a production-grade, config-driven pipeline with:
- Data cleaning and feature engineering matching your notebook (customerID drop, TotalCharges coercion/median fill, service text normalization, Yes/No to 1/0, tenure_group, LongTerm, Contract ordinal, AvailingInternetService, NumServices, AvgCharge, optional Yeo-Johnson)
- Preprocessing with One-Hot encoding for categoricals and passthrough numerics
- Model training with cross-validation
- Hyperparameter tuning (GridSearchCV)
- Evaluation with metrics and plots
- Soft-voting ensembling
- Best-model selection

## Install
```bash
pip install -r requirements.txt
```

## Configure
Edit `config/config.yaml`:
- Set `data.path` to your raw CSV (e.g., `data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv`)
- Confirm `data.target` (default: `Churn`)
- Toggle feature engineering and power transform under `preprocessing`
- Choose models and their params; add grids under `tuning.param_grids`

## Run
- Full run (prepare, tune if enabled, train, evaluate, ensemble, pick best)
```bash
python src/churn_pipeline.py --config config/config.yaml --mode full
```

- Tune only
```bash
python src/churn_pipeline.py --config config/config.yaml --mode tune
```

- Train only (no tuning)
```bash
python src/churn_pipeline.py --config config/config.yaml --mode train
```

- Evaluate saved models
```bash
python src/churn_pipeline.py --config config/config.yaml --mode eval
```

- Pick best model (by metric; default `roc_auc`)
```bash
python src/churn_pipeline.py --config config/config.yaml --mode pick-best --metric f1
```

## Artifacts
- Processed data: `artifacts/data/processed/`
- Models: `artifacts/models/` (`*.joblib`, CV summaries, best params and grid results)
- Reports: `artifacts/reports/` (per-model `*_metrics.json`, classification reports) and `artifacts/reports/figures/` (CM/ROC/PR plots)
- Summary: `artifacts/reports/summary_metrics.json`
- Best model copy: `artifacts/models/best_model.joblib`




telecom_churn/
├── data/
│   ├── raw_data.csv             # Raw data
│   └── processed_data.csv       # Preprocessed data
│
├── src/
│   ├── data_preprocessing.py    # Preprocessing and feature engineering
│   ├── model_training.py        # Individual model training
│   ├── model_evaluation.py      # Evaluation of multiple models
│   ├── model_ensembling.py     # Ensemble methods (Voting, Stacking, etc.)
│   ├── churn_pipeline.py        # Main pipeline for end-to-end workflow
│   └── hyperparameter_tuning.py # Hyperparameter optimization (GridSearchCV, RandomizedSearchCV)
│
├── notebooks/
│   └── churn_prediction.ipynb   # Exploration, EDA, and model comparison
│
├── requirements.txt            # Python dependencies
└── README.md                   # Project overview
`
