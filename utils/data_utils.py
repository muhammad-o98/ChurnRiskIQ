"""
Data processing utilities for the Streamlit app
"""

from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, PowerTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Complete data preprocessing pipeline"""
    df = data.copy()
    
    # Drop customerID if exists
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
    
    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Impute missing TotalCharges
    if df['TotalCharges'].isna().sum() > 0:
        median_total = df['TotalCharges'].median()
        df['TotalCharges'] = df['TotalCharges'].fillna(median_total)
    
    # Replace service values
    replace_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for col in replace_cols:
        if col in df.columns:
            df[col] = df[col].replace({
                'No internet service': 'No',
                'No phone service': 'No'
            })
    
    # Encode boolean columns
    boolean_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'Churn'
    ]
    
    for col in boolean_cols:
        if col in df.columns:
            if set(df[col].unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map({'Yes': 1, 'No': 0})
                df[col] = df[col].astype(int)
    
    # Encode Contract (ordinal)
    if 'Contract' in df.columns:
        contract_order = {
            'Month-to-month': 1,
            'One year': 2,
            'Two year': 3
        }
        df['Contract'] = df['Contract'].map(contract_order)
    
    return df


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create engineered features"""
    df = data.copy()
    
    # Tenure group
    def tenure_bin(tenure):
        if tenure <= 12:
            return '0-12 Months'
        elif tenure <= 24:
            return '13-24 Months'
        elif tenure <= 48:
            return '25-48 Months'
        elif tenure <= 60:
            return '49-60 Months'
        else:
            return '61-72 Months'
    
    df['tenure_group'] = df['tenure'].apply(tenure_bin)
    
    # LongTerm customer flag
    df['LongTerm'] = (df['tenure'] > 24).astype(int)
    
    # AvailingInternetService
    df['AvailingInternetService'] = df['InternetService'].apply(
        lambda x: 0 if x == 'No' else 1
    )
    
    # NumServices
    service_cols = [
        'PhoneService', 'MultipleLines', 'AvailingInternetService',
        'OnlineSecurity', 'OnlineBackup', 'TechSupport',
        'StreamingTV', 'StreamingMovies'
    ]
    df['NumServices'] = df[service_cols].sum(axis=1)
    
    # AvgCharge
    df['AvgCharge'] = df['MonthlyCharges'] / df['NumServices'].replace(0, 1)
    
    # LessThan6Months
    df['LessThan6Months'] = (df['tenure'] < 6).astype(int)
    
    return df


def split_data(data: pd.DataFrame, target_col: str = 'Churn', test_size: float = 0.2, random_state: int = 42):
    """Split data into train and test sets"""
    X = data.drop(columns=[target_col])
    y = data[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    
    return X_train, X_test, y_train, y_test


def get_preprocessor(X_train: pd.DataFrame):
    """Get sklearn preprocessor for model training"""
    categorical_features = ['gender', 'InternetService', 'PaymentMethod', 'tenure_group']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'AvgCharge']
    boolean_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'LessThan6Months',
        'AvailingInternetService', 'LongTerm'
    ]
    
    onehot = OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')
    
    preprocessor = ColumnTransformer([
        ('cat', onehot, categorical_features),
        ('num', 'passthrough', numeric_features),
        ('bool', 'passthrough', boolean_cols)
    ])
    
    return preprocessor


def get_feature_names_from_preprocessor(preprocessor, X_train):
    """Extract feature names after preprocessing"""
    categorical_features = ['gender', 'InternetService', 'PaymentMethod', 'tenure_group']
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'NumServices', 'AvgCharge']
    boolean_cols = [
        'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
        'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'PaperlessBilling', 'LessThan6Months',
        'AvailingInternetService', 'LongTerm'
    ]
    
    cat_transformer = preprocessor.named_transformers_['cat']
    cat_feature_names = cat_transformer.get_feature_names_out(categorical_features).tolist()
    
    feature_names = cat_feature_names + numeric_features + boolean_cols
    return feature_names


# ---------- Path helpers & sample data loaders ----------
def get_project_root() -> Path:
    """Return absolute Path to the project root folder.

    utils/ lives directly under the project root, so parent of this file's dir is root.
    """
    return Path(__file__).resolve().parents[1]


def get_data_path(*parts: str) -> Path:
    """Build an absolute path under the project's data/ directory.

    Example: get_data_path('churndata.csv') -> <root>/data/churndata.csv
    """
    return get_project_root() / 'data' / Path(*parts)


def load_sample_dataset(filename: str = 'churndata.csv') -> pd.DataFrame:
    """Load a sample dataset from the data/ directory with robust path resolution.

    Raises FileNotFoundError with a friendly message if the file is missing.
    """
    file_path = get_data_path(filename)
    if not file_path.exists():
        raise FileNotFoundError(f"Sample dataset not found at {file_path}")
    return pd.read_csv(file_path)
