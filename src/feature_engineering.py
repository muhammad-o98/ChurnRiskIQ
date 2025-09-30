import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class TelcoFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Robust feature engineering:
    - Drop 'customerID' if present
    - Coerce 'TotalCharges' to numeric and fill NaNs with median
    - Normalize 'No internet service'/'No phone service' to 'No'
    - Map Yes/No to 1/0 for boolean-like columns
    - tenure_group bins and LongTerm flag
    - Contract ordinal map
    - AvailingInternetService indicator
    - NumServices: robust numeric sum of service flags (avoids mixed int/str TypeError)
    - AvgCharge safe divide
    - Optional Yeo-Johnson on selected numerics
    """
    service_cols_default = [
        "PhoneService", "MultipleLines", "AvailingInternetService",
        "OnlineSecurity", "OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"
    ]

    def __init__(
        self,
        replace_cols: Optional[List[str]] = None,
        boolean_cols: Optional[List[str]] = None,
        contract_mapping: Optional[Dict[str, int]] = None,
        power_transform_numeric: bool = True,
        power_transform_features: Optional[List[str]] = None,
        power_transform_method: str = "yeo-johnson",
        copy: bool = True,
    ):
        self.replace_cols = replace_cols or [
            "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        self.boolean_cols = boolean_cols or [
            "SeniorCitizen", "Partner", "Dependents", "PhoneService", "MultipleLines",
            "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
            "StreamingTV", "StreamingMovies", "PaperlessBilling", "Churn"
        ]
        self.contract_mapping = contract_mapping or {
            "Month-to-month": 1,
            "One year": 2,
            "Two year": 3
        }
        self.power_transform_numeric = power_transform_numeric
        self.power_transform_features = power_transform_features or [
            "tenure", "MonthlyCharges", "TotalCharges", "NumServices", "AvgCharge"
        ]
        self.power_transform_method = power_transform_method
        self.copy = copy

        self._pt: Optional[PowerTransformer] = None
        self.fitted_: bool = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None):
        X_tmp = X.copy() if self.copy else X

        if "TotalCharges" in X_tmp.columns:
            X_tmp["TotalCharges"] = pd.to_numeric(X_tmp["TotalCharges"], errors="coerce")

        X_tmp = self._basic_transform(X_tmp)

        if self.power_transform_numeric:
            self._pt = PowerTransformer(method=self.power_transform_method, standardize=False)
            cols = [c for c in self.power_transform_features if c in X_tmp.columns]
            if cols:
                X_pt = X_tmp[cols].copy()
                for c in cols:
                    if X_pt[c].isna().any():
                        X_pt[c] = X_pt[c].fillna(X_pt[c].median())
                self._pt.fit(X_pt.values)
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        assert self.fitted_, "TelcoFeatureBuilder must be fit before transform."
        df = X.copy() if self.copy else X

        if "customerID" in df.columns:
            df = df.drop(columns=["customerID"])

        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            if df["TotalCharges"].isna().any():
                df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

        # Normalize text before mapping
        self._normalize_service_text(df, self.replace_cols + self.service_cols_default)
        # Coerce flags to numeric
        self._coerce_yes_no_flags(df, self.boolean_cols + self.service_cols_default)

        if "tenure" in df.columns:
            df["tenure_group"] = df["tenure"].apply(self._tenure_bin).astype("object")
            df["LongTerm"] = (df["tenure"] > 24).astype(int)

        if "Contract" in df.columns and df["Contract"].dtype == object:
            df["Contract"] = df["Contract"].map(self.contract_mapping).astype("Int64").fillna(0).astype(int)

        if "InternetService" in df.columns:
            df["AvailingInternetService"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)

        present = [c for c in self.service_cols_default if c in df.columns]
        if present:
            # Force numeric and safe sum
            df["NumServices"] = df[present].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum(axis=1)

        if "MonthlyCharges" in df.columns and "NumServices" in df.columns:
            denom = df["NumServices"].replace(0, np.nan)
            df["AvgCharge"] = df["MonthlyCharges"] / denom
            df["AvgCharge"] = df["AvgCharge"].fillna(df["MonthlyCharges"])

        if self.power_transform_numeric and self._pt is not None:
            cols = [c for c in self.power_transform_features if c in df.columns]
            if cols:
                X_pt = df[cols].copy()
                for c in cols:
                    if X_pt[c].isna().any():
                        X_pt[c] = X_pt[c].fillna(X_pt[c].median())
                transformed = self._pt.transform(X_pt.values)
                for i, c in enumerate(cols):
                    df[c] = transformed[:, i]

        return df

    @staticmethod
    def _tenure_bin(tenure: Any) -> str:
        try:
            t = float(tenure)
        except Exception:
            return "Unknown"
        if t <= 12:
            return "0-12 Months"
        elif t <= 24:
            return "13-24 Months"
        elif t <= 48:
            return "25-48 Months"
        elif t <= 60:
            return "49-60 Months"
        else:
            return "61-72 Months"

    def _normalize_service_text(self, df: pd.DataFrame, cols: List[str]) -> None:
        for col in cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

    def _coerce_yes_no_flags(self, df: pd.DataFrame, cols: List[str]) -> None:
        for col in cols:
            if col not in df.columns:
                continue
            s = df[col]
            if s.dtype == object:
                s = s.replace({"No internet service": "No", "No phone service": "No"})
                s_lower = s.astype(str).str.lower()
                mapping = {"yes": 1, "no": 0, "true": 1, "false": 0}
                mapped = s_lower.map(mapping)
                s = mapped.where(~mapped.isna(), s)
            if s.dtype == bool:
                s = s.astype(int)
            s = pd.to_numeric(s, errors="coerce").fillna(0).astype(int)
            df[col] = s

    def _basic_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "InternetService" in out.columns and "AvailingInternetService" not in out.columns:
            out["AvailingInternetService"] = out["InternetService"].apply(lambda x: 0 if x == "No" else 1)
        self._normalize_service_text(out, self.replace_cols + self.service_cols_default)
        present = [c for c in self.service_cols_default if c in out.columns]
        if present:
            self._coerce_yes_no_flags(out, present)
            out["NumServices"] = out[present].apply(pd.to_numeric, errors="coerce").fillna(0).astype(int).sum(axis=1)
        if "MonthlyCharges" in out.columns and "NumServices" in out.columns and "AvgCharge" not in out.columns:
            denom = out["NumServices"].replace(0, np.nan)
            out["AvgCharge"] = out["MonthlyCharges"] / denom
            out["AvgCharge"] = out["AvgCharge"].fillna(out["MonthlyCharges"])
        return out