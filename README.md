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