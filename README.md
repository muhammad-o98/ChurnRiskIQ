# Telecom Churn Prediction - Complete ML & Analytics Platform

An end-to-end machine learning platform for telecom customer churn prediction, featuring:
- **Interactive Streamlit Web Application** with modern UI/UX
- **6 ML Models**: Decision Tree, Random Forest, Gradient Boosting, XGBoost, LightGBM, CatBoost
- **SHAP Analysis** for model explainability
- **Risk Segmentation** with retention strategies
- **Executive Dashboard** with KPIs and insights

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Streamlit App
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Application Features

### 🏠 **Home Page**
- Project overview and navigation
- Quick metrics dashboard
- Feature highlights

### 📤 **Data Upload & Preprocessing**
- Upload your own CSV or use sample data
- Automated data cleaning and preprocessing
- Feature engineering (6 new features)
- Interactive visualizations (distributions, correlations)

### 🤖 **Model Training**
- Train multiple models simultaneously
- Real-time progress tracking
- Model comparison with 10+ metrics
- ROC curves and confusion matrices

### 🔍 **SHAP Analysis**
- Feature importance visualization
- SHAP summary plots
- Top churn drivers analysis
- Supports both TreeExplainer and KernelExplainer

### ⚠️ **Risk Segmentation**
- Customer segmentation (High/Medium/Low risk)
- Churn probability scoring
- Retention strategies per segment
- Export customer lists

### 📊 **Executive Dashboard**
- KPIs and metrics overview
- Model performance summary
- Risk distribution insights
- Executive summary report
- Export capabilities

---

## 📁 Project Structure

```
telecom-churn/
├── app.py                          # Main Streamlit application
├── pages/                          # Multi-page app structure
│   ├── 1_📤_Data_Upload.py        # Data upload and preprocessing
│   ├── 2_🤖_Model_Training.py     # Model training and comparison
│   ├── 3_🔍_SHAP_Analysis.py      # SHAP explainability
│   ├── 4_⚠️_Risk_Segmentation.py  # Customer risk segments
│   └── 5_📊_Dashboard.py          # Executive dashboard
├── utils/                          # Utility modules
│   ├── session_state.py           # Session state management
│   ├── data_utils.py              # Data preprocessing
│   └── model_utils.py             # Model training & evaluation
├── data/
│   └── churndata.csv              # Sample dataset
├── notebooks/
│   └── telecom_churn.ipynb        # Jupyter notebook analysis
└── src/                           # CLI pipeline (optional)
    ├── main.py
    ├── churn_pipeline.py
    └── ...
```

---

## 🔧 Configuration-Driven CLI Pipeline

For advanced users, a config-driven CLI pipeline is also available:

### Configure
Edit [config/config.yaml](config/config.yaml):
- data.path: path to your CSV
- models.base_models: select models to run
- tuning.param_grids: hyperparameter grids
- tuning.strategy: "grid" or "random"

### Run Modes
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
