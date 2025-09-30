import os
import json
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from feature_engineering import TelcoFeatureBuilder


@dataclass
class PreprocessingConfig:
    enable_feature_engineering: bool = True
    power_transform_numeric: bool = True
    power_transform_features: Optional[List[str]] = None

    categorical_features: Optional[List[str]] = None
    numerical_features: Optional[List[str]] = None

    impute_categorical: str = "most_frequent"
    impute_numeric: Optional[str] = None
    onehot_sparse_output: bool = False
    onehot_drop_first: bool = True  # align with notebook


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


def load_data(csv_path: str, logger: Optional[logging.Logger] = None) -> pd.DataFrame:
    if logger:
        logger.info(f"Loading data from {csv_path}")
    df = pd.read_csv(csv_path)
    if logger:
        logger.info(f"Loaded dataframe with shape {df.shape}")
    return df


def build_preprocessor(
    fe: TelcoFeatureBuilder,
    numeric_features: List[str],
    categorical_features: List[str],
    cfg: PreprocessingConfig,
) -> Pipeline:
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy=cfg.impute_categorical)),
            ("encoder", OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=cfg.onehot_sparse_output,
                drop="first" if cfg.onehot_drop_first else None
            )),
        ]
    )

    num_pipe = Pipeline(steps=[("imputer", SimpleImputer(strategy=cfg.impute_numeric))]) if cfg.impute_numeric else "passthrough"

    ct = ColumnTransformer(
        transformers=[
            ("cat", cat_pipe, categorical_features),
            ("num", num_pipe, numeric_features),
        ]
    )

    preprocessor = Pipeline(
        steps=[
            ("fe", fe),
            ("ct", ct),
        ]
    )
    return preprocessor


def save_processed_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: pd.Series,
    y_test: pd.Series,
    feature_names: List[str],
    out_dir: str,
    logger: Optional[logging.Logger] = None,
) -> None:
    ensure_dir(out_dir)
    if logger:
        logger.info(f"Saving processed data to {out_dir}")
    np.savez_compressed(os.path.join(out_dir, "X_train.npz"), X_train=X_train)
    np.savez_compressed(os.path.join(out_dir, "X_test.npz"), X_test=X_test)
    y_train.to_csv(os.path.join(out_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(out_dir, "y_test.csv"), index=False)
    with open(os.path.join(out_dir, "features.json"), "w") as f:
        json.dump(feature_names, f, indent=2)


def prepare_data(
    data_path: str,
    target: str,
    test_size: float,
    random_state: int,
    stratify: bool = True,
    preprocessing_cfg: Optional[Dict[str, Any]] = None,
    processed_out_dir: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
):
    cfg = PreprocessingConfig(**(preprocessing_cfg or {}))
    df = load_data(data_path, logger)

    y_raw = df[target]
    if y_raw.dtype == object:
        y = y_raw.map({"Yes": 1, "No": 0}).astype("Int64").fillna(0).astype(int)
    else:
        y = y_raw
    X = df.drop(columns=[target])

    stratify_arr = y if stratify else None
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_arr
    )

    fe = TelcoFeatureBuilder(
        power_transform_numeric=cfg.power_transform_numeric,
        power_transform_features=cfg.power_transform_features,
    )
    fe.fit(X_train, y_train)

    X_train_fe = fe.transform(X_train)
    X_test_fe = fe.transform(X_test)

    if cfg.categorical_features is not None:
        categorical_features = [c for c in cfg.categorical_features if c in X_train_fe.columns]
    else:
        categorical_features = X_train_fe.select_dtypes(include=["object", "category"]).columns.tolist()

    if cfg.numerical_features is not None:
        numeric_features = [c for c in cfg.numerical_features if c in X_train_fe.columns]
    else:
        numeric_features = [c for c in X_train_fe.columns if c not in categorical_features]

    preprocessor = build_preprocessor(
        fe=fe,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        cfg=cfg,
    )

    ct = preprocessor.named_steps["ct"]
    ct.fit(X_train_fe)
    X_train_trans = ct.transform(X_train_fe)
    X_test_trans = ct.transform(X_test_fe)

    feature_names: List[str] = []
    if "cat" in ct.named_transformers_:
        cat = ct.named_transformers_["cat"]
        enc = cat.named_steps["encoder"]
        feature_names.extend(enc.get_feature_names_out(categorical_features).tolist())
    feature_names.extend(numeric_features)

    if processed_out_dir:
        save_processed_data(X_train_trans, X_test_trans, y_train, y_test, feature_names, processed_out_dir, logger)

    roles = {"categorical": categorical_features, "numerical": numeric_features}
    return preprocessor, X_train_trans, y_train, X_test_trans, y_test, feature_names, roles