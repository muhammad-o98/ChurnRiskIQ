import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import PowerTransformer


class TelcoFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Feature engineering aligned with notebook logic:
    - Drop 'customerID' if present
    - Coerce 'TotalCharges' to numeric and fill NaNs with median
    - Normalize 'No internet service'/'No phone service' to 'No'
    - Map Yes/No to 1/0 for boolean-like columns
    - tenure_group bins: [0-12,13-24,25-48,49-60,61-72]
    - LongTerm: 1 if tenure > 24 else 0
    - Contract ordinal mapping: {'Month-to-month':1,'One year':2,'Two year':3}
    - AvailingInternetService: 0 if InternetService=='No' else 1
    - NumServices: sum of service flags
    - AvgCharge: MonthlyCharges / NumServices (safe; if NumServices==0, equals MonthlyCharges)
    - Optional Yeo-Johnson transform for selected numeric features
    """
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

        X_tmp = self._basic_transform(X_tmp, fit_mode=True)

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

        for col in self.replace_cols:
            if col in df.columns and df[col].dtype == object:
                df[col] = df[col].replace({"No internet service": "No", "No phone service": "No"})

        for col in self.boolean_cols:
            if col in df.columns and df[col].dtype == object:
                uniques = set(pd.Series(df[col].dropna().unique(), dtype=object))
                if uniques == {"Yes", "No"} or uniques == {"No", "Yes"}:
                    df[col] = df[col].map({"Yes": 1, "No": 0}).astype("Int64").fillna(0).astype(int)

        if "tenure" in df.columns:
            df["tenure_group"] = df["tenure"].apply(self._tenure_bin).astype("object")
            df["LongTerm"] = (df["tenure"] > 24).astype(int)

        if "Contract" in df.columns:
            if df["Contract"].dtype == object:
                df["Contract"] = df["Contract"].map(self.contract_mapping).astype("Int64").fillna(0).astype(int)

        if "InternetService" in df.columns:
            df["AvailingInternetService"] = df["InternetService"].apply(lambda x: 0 if x == "No" else 1)

        service_cols = [
            "PhoneService", "MultipleLines", "AvailingInternetService",
            "OnlineSecurity", "OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        present = [c for c in service_cols if c in df.columns]
        for c in present:
            if df[c].dtype == object:
                uniq = set(pd.Series(df[c].dropna().unique(), dtype=object))
                if uniq == {"Yes", "No"} or uniq == {"No", "Yes"}:
                    df[c] = df[c].map({"Yes": 1, "No": 0}).astype("Int64").fillna(0).astype(int)
        if present:
            df["NumServices"] = df[present].sum(axis=1)

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

    def _basic_transform(self, df: pd.DataFrame, fit_mode: bool = False) -> pd.DataFrame:
        out = df.copy()
        if "InternetService" in out.columns and "AvailingInternetService" not in out.columns:
            out["AvailingInternetService"] = out["InternetService"].apply(lambda x: 0 if x == "No" else 1)
        service_cols = [
            "PhoneService", "MultipleLines", "AvailingInternetService",
            "OnlineSecurity", "OnlineBackup", "TechSupport", "StreamingTV", "StreamingMovies"
        ]
        present = [c for c in service_cols if c in out.columns]
        for c in present:
            if out[c].dtype == object:
                uniq = set(pd.Series(out[c].dropna().unique(), dtype=object))
                if uniq == {"Yes", "No"} or uniq == {"No", "Yes"}:
                    out[c] = out[c].map({"Yes": 1, "No": 0}).astype("Int64").fillna(0).astype(int)
        if present and "NumServices" not in out.columns:
            out["NumServices"] = out[present].sum(axis=1)
        if "MonthlyCharges" in out.columns and "NumServices" in out.columns and "AvgCharge" not in out.columns:
            denom = out["NumServices"].replace(0, np.nan)
            out["AvgCharge"] = out["MonthlyCharges"] / denom
            out["AvgCharge"] = out["AvgCharge"].fillna(out["MonthlyCharges"])
        return out