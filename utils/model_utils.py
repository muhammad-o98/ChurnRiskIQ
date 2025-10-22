"""
Model training and evaluation utilities
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_model_registry():
    """Return dictionary of available models"""
    return {
        'dt': {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(
                class_weight='balanced',
                criterion='gini',
                max_depth=5,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=42
            )
        },
        'rf': {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                class_weight='balanced',
                criterion='gini',
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                max_features='sqrt',
                n_estimators=100,  # Reduced for speed
                bootstrap=True,
                n_jobs=-1,
                random_state=42
            )
        },
        'gb': {
            'name': 'Gradient Boosting',
            'model': GradientBoostingClassifier(
                loss='log_loss',
                learning_rate=0.05,
                n_estimators=100,  # Reduced for speed
                criterion='friedman_mse',
                min_samples_split=10,
                min_samples_leaf=5,
                max_depth=3,
                random_state=42
            )
        },
        'xgb': {
            'name': 'XGBoost',
            'model': XGBClassifier(
                objective='binary:logistic',
                learning_rate=0.01,
                max_depth=3,
                min_child_weight=1,
                subsample=0.7,
                colsample_bytree=0.7,
                n_estimators=100,  # Reduced for speed
                eval_metric='logloss',
                use_label_encoder=False,
                n_jobs=-1,
                random_state=42
            )
        },
        'lgbm': {
            'name': 'LightGBM',
            'model': LGBMClassifier(
                objective='binary',
                learning_rate=0.01,
                n_estimators=100,  # Reduced for speed
                max_depth=10,
                num_leaves=15,
                min_child_samples=20,
                reg_alpha=0.1,
                reg_lambda=0.5,
                n_jobs=-1,
                random_state=42,
                verbose=-1
            )
        },
        'cat': {
            'name': 'CatBoost',
            'model': CatBoostClassifier(
                iterations=100,  # Reduced for speed
                learning_rate=0.05,
                depth=4,
                l2_leaf_reg=3,
                verbose=0,
                random_seed=42
            )
        }
    }


def train_model(model_key, preprocessor, X_train, y_train):
    """Train a single model"""
    registry = get_model_registry()
    clf = registry[model_key]['model']
    
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', clf)
    ])
    
    pipeline.fit(X_train, y_train)
    return pipeline


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """Evaluate model performance"""
    # Get predictions
    y_train_probs = model.predict_proba(X_train)[:, 1]
    y_test_probs = model.predict_proba(X_test)[:, 1]
    
    y_train_pred = (y_train_probs >= 0.5).astype(int)
    y_test_pred = (y_test_probs >= 0.5).astype(int)
    
    # Calculate metrics
    train_metrics = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred),
        'recall': recall_score(y_train, y_train_pred),
        'f1': f1_score(y_train, y_train_pred),
        'roc_auc': roc_auc_score(y_train, y_train_probs),
        'pr_auc': average_precision_score(y_train, y_train_probs),
        'brier': brier_score_loss(y_train, y_train_probs)
    }
    
    test_metrics = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred),
        'recall': recall_score(y_test, y_test_pred),
        'f1': f1_score(y_test, y_test_pred),
        'roc_auc': roc_auc_score(y_test, y_test_probs),
        'pr_auc': average_precision_score(y_test, y_test_probs),
        'brier': brier_score_loss(y_test, y_test_probs)
    }
    
    # Calculate best threshold based on PR curve
    precision, recall, thresholds = precision_recall_curve(y_test, y_test_probs)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'best_threshold': best_threshold,
        'confusion_matrix': cm,
        'y_test_probs': y_test_probs,
        'y_test_pred': y_test_pred
    }


def compare_models(results_dict):
    """Compare multiple models and return summary DataFrame"""
    comparison = []
    
    for model_name, results in results_dict.items():
        test_metrics = results['test_metrics']
        comparison.append({
            'Model': model_name,
            'Accuracy': test_metrics['accuracy'],
            'Precision': test_metrics['precision'],
            'Recall': test_metrics['recall'],
            'F1 Score': test_metrics['f1'],
            'ROC AUC': test_metrics['roc_auc'],
            'PR AUC': test_metrics['pr_auc'],
            'Brier Score': test_metrics['brier']
        })
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('PR AUC', ascending=False)
    return df


def get_best_model(results_dict, metric='pr_auc'):
    """Get the best performing model based on specified metric"""
    best_score = -np.inf
    best_model_name = None
    
    for model_name, results in results_dict.items():
        score = results['test_metrics'][metric]
        if score > best_score:
            best_score = score
            best_model_name = model_name
    
    return best_model_name, best_score
