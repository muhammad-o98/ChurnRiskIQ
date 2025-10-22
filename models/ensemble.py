"""Ensemble training utilities: voting, stacking, calibration, CV, Optuna, imbalance handling.

This module augments the existing utils.model_utils with advanced training paths.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline as SkPipeline
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.utils.class_weight import compute_class_weight

try:
    import optuna
except Exception:  # pragma: no cover
    optuna = None

from utils.model_utils import get_model_registry, evaluate_model


@dataclass
class AdvancedTrainConfig:
    cv_splits: int = 5
    random_state: int = 42
    use_smote: bool = False
    smote_kind: str = "SMOTE"  # or "ADASYN"
    use_optuna: bool = False
    n_trials: int = 20
    calibrate: bool = True
    ensemble_voting: bool = True
    ensemble_stacking: bool = True
    scoring: str = "pr_auc"  # currently only pr_auc in this module


def _get_resampler(kind: str):
    from imblearn.over_sampling import SMOTE, ADASYN
    if kind.upper() == "SMOTE":
        return SMOTE(random_state=42)
    if kind.upper() == "ADASYN":
        return ADASYN(random_state=42)
    raise ValueError("Unsupported resampler: %s" % kind)


def _maybe_optuna_tune(model_key: str, X: pd.DataFrame, y: pd.Series, n_trials: int = 20, cv_splits: int = 3,
                       random_state: int = 42):
    """Very small Optuna tuner for a couple of key models; returns a configured estimator.
    Falls back to the default model if optuna is unavailable.
    """
    from utils.model_utils import get_model_registry
    reg = get_model_registry()
    base = reg[model_key]['model']

    if optuna is None:
        return base

    def objective(trial):
        clf = None
        if model_key == 'xgb':
            from xgboost import XGBClassifier
            clf = XGBClassifier(
                objective='binary:logistic',
                eval_metric='logloss',
                n_estimators=trial.suggest_int('n_estimators', 150, 400),
                max_depth=trial.suggest_int('max_depth', 3, 8),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.5, 1.0),
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_key == 'lgbm':
            from lightgbm import LGBMClassifier
            clf = LGBMClassifier(
                objective='binary',
                n_estimators=trial.suggest_int('n_estimators', 200, 600),
                num_leaves=trial.suggest_int('num_leaves', 15, 63),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        else:
            clf = base

        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        probs = cross_val_predict(clf, X, y, cv=skf, method='predict_proba')[:, 1]
        return average_precision_score(y, probs)

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = study.best_params
    # Recreate the estimator with best params for known models
    if model_key == 'xgb':
        from xgboost import XGBClassifier
        tuned = XGBClassifier(objective='binary:logistic', eval_metric='logloss', n_jobs=-1, random_state=random_state, **best_params)
    elif model_key == 'lgbm':
        from lightgbm import LGBMClassifier
        tuned = LGBMClassifier(objective='binary', n_jobs=-1, random_state=random_state, verbose=-1, **best_params)
    else:
        tuned = base
    return tuned


def train_advanced(
    model_keys: List[str],
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    config: Optional[AdvancedTrainConfig] = None,
):
    """Train advanced models with CV, optional Optuna, ensembles and calibration.

    Returns: best_pipeline, results_dict (including ensembles), comparison_df
    """
    if config is None:
        config = AdvancedTrainConfig()

    registry = get_model_registry()

    # Prepare estimators (optionally tuned)
    estimators = []
    for k in model_keys:
        base = registry[k]['model']
        clf = _maybe_optuna_tune(k, X_train, y_train, n_trials=config.n_trials, cv_splits=max(3, config.cv_splits//2)) if config.use_optuna else base
        estimators.append((k, clf))

    # Build pipelines per estimator (with optional resampling)
    results: Dict[str, dict] = {}

    for key, est in estimators:
        if config.use_smote:
            resampler = _get_resampler(config.smote_kind)
            pipe = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resample', resampler),
                ('classifier', est),
            ])
        else:
            pipe = SkPipeline([
                ('preprocessor', preprocessor),
                ('classifier', est),
            ])
        pipe.fit(X_train, y_train)
        res = evaluate_model(pipe, X_train, y_train, X_test, y_test)
        results[key] = res

    # Ensemble - Voting
    voting_pipe = None
    if config.ensemble_voting and len(estimators) >= 2:
        voting = VotingClassifier(estimators=estimators, voting='soft', n_jobs=-1)
        if config.use_smote:
            resampler = _get_resampler(config.smote_kind)
            voting_pipe = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resample', resampler),
                ('classifier', voting),
            ])
        else:
            voting_pipe = SkPipeline([
                ('preprocessor', preprocessor),
                ('classifier', voting),
            ])
        voting_pipe.fit(X_train, y_train)
        results['voting_ensemble'] = evaluate_model(voting_pipe, X_train, y_train, X_test, y_test)

    # Ensemble - Stacking
    stacking_pipe = None
    if config.ensemble_stacking and len(estimators) >= 2:
        stk = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=200), stack_method='predict_proba', cv=config.cv_splits, n_jobs=-1)
        if config.use_smote:
            resampler = _get_resampler(config.smote_kind)
            stacking_pipe = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resample', resampler),
                ('classifier', stk),
            ])
        else:
            stacking_pipe = SkPipeline([
                ('preprocessor', preprocessor),
                ('classifier', stk),
            ])
        stacking_pipe.fit(X_train, y_train)
        results['stacking_ensemble'] = evaluate_model(stacking_pipe, X_train, y_train, X_test, y_test)

    # Calibration on the best of ensembles or best base
    best_name = max(results.keys(), key=lambda k: results[k]['test_metrics']['pr_auc'])
    best_pipe = {'voting_ensemble': voting_pipe, 'stacking_ensemble': stacking_pipe}.get(best_name)
    if best_pipe is None:
        # It's a base model pipeline; rebuild calibrated variant
        base_key = best_name
        base_est = dict(estimators)[base_key]
        cal = CalibratedClassifierCV(base_est, method='isotonic', cv=3) if config.calibrate else base_est
        if config.use_smote:
            resampler = _get_resampler(config.smote_kind)
            best_pipe = ImbPipeline([
                ('preprocessor', preprocessor),
                ('resample', resampler),
                ('classifier', cal),
            ])
        else:
            best_pipe = SkPipeline([
                ('preprocessor', preprocessor),
                ('classifier', cal),
            ])
        best_pipe.fit(X_train, y_train)
        results[base_key + ('_cal' if config.calibrate else '')] = evaluate_model(best_pipe, X_train, y_train, X_test, y_test)
        best_name = base_key + ('_cal' if config.calibrate else '')
    else:
        if config.calibrate:
            cal = CalibratedClassifierCV(best_pipe.named_steps['classifier'], method='isotonic', cv=3)
            # Rebuild pipeline with same steps, replacing classifier
            steps = list(best_pipe.steps)
            steps[-1] = ('classifier', cal)
            best_pipe = type(best_pipe)(steps)
            best_pipe.fit(X_train, y_train)
            results[best_name + '_cal'] = evaluate_model(best_pipe, X_train, y_train, X_test, y_test)
            best_name = best_name + '_cal'

    # Build comparison df
    comp_rows = []
    for name, res in results.items():
        tm = res['test_metrics']
        comp_rows.append({
            'Model': name,
            'Accuracy': tm['accuracy'],
            'Precision': tm['precision'],
            'Recall': tm['recall'],
            'F1-Score': tm['f1'],
            'ROC AUC': tm['roc_auc'],
            'PR AUC': tm['pr_auc'],
            'Brier Score': tm['brier'],
        })
    comparison_df = pd.DataFrame(comp_rows).sort_values('PR AUC', ascending=False)

    return best_pipe, results, comparison_df
