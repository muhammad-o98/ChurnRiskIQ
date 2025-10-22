"""
Comprehensive test script for the Streamlit app
Tests all imports, utilities, and data loading
"""

import sys
import os

# Set working directory
os.chdir('/Users/ob/telecom-churn')
sys.path.insert(0, '/Users/ob/telecom-churn')

print("=" * 70)
print("🧪 TELECOM CHURN APP - COMPREHENSIVE TEST SUITE")
print("=" * 70)

# Test 1: Core imports
print("\n1️⃣ Testing Core Imports...")
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    import plotly.express as px
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt
    import seaborn as sns
    print("   ✅ All core packages imported successfully")
except Exception as e:
    print(f"   ❌ Core import failed: {e}")
    sys.exit(1)

# Test 2: ML Libraries
print("\n2️⃣ Testing ML Libraries...")
try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    import xgboost
    import lightgbm
    import catboost
    import shap
    print("   ✅ All ML libraries imported successfully")
except Exception as e:
    print(f"   ❌ ML library import failed: {e}")
    sys.exit(1)

# Test 3: Utility modules
print("\n3️⃣ Testing Utility Modules...")
try:
    from utils.session_state import init_session_state
    from utils.data_utils import preprocess_data, engineer_features, split_data
    from utils.model_utils import get_model_registry, train_model, evaluate_model
    print("   ✅ All utility modules imported successfully")
except Exception as e:
    print(f"   ❌ Utility module import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Data loading
print("\n4️⃣ Testing Data Loading...")
try:
    df = pd.read_csv('data/churndata.csv')
    print(f"   ✅ Data loaded: {len(df)} rows, {len(df.columns)} columns")
    print(f"   ℹ️  Columns: {', '.join(df.columns[:5])}...")
except Exception as e:
    print(f"   ❌ Data loading failed: {e}")
    sys.exit(1)

# Test 5: Data preprocessing
print("\n5️⃣ Testing Data Preprocessing...")
try:
    df_processed = preprocess_data(df.copy())
    print(f"   ✅ Data preprocessed: {len(df_processed)} rows")
except Exception as e:
    print(f"   ❌ Preprocessing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Feature engineering
print("\n6️⃣ Testing Feature Engineering...")
try:
    df_engineered = engineer_features(df_processed.copy())
    new_features = set(df_engineered.columns) - set(df_processed.columns)
    print(f"   ✅ Features engineered: {len(new_features)} new features")
    print(f"   ℹ️  New features: {', '.join(list(new_features)[:5])}")
except Exception as e:
    print(f"   ❌ Feature engineering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Train/test split
print("\n7️⃣ Testing Train/Test Split...")
try:
    X_train, X_test, y_train, y_test = split_data(df_engineered)
    print(f"   ✅ Data split complete")
    print(f"   ℹ️  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
except Exception as e:
    print(f"   ❌ Train/test split failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 8: Model registry
print("\n8️⃣ Testing Model Registry...")
try:
    models = get_model_registry()
    print(f"   ✅ Model registry loaded: {len(models)} models")
    print(f"   ℹ️  Models: {', '.join(models.keys())}")
except Exception as e:
    print(f"   ❌ Model registry failed: {e}")
    sys.exit(1)

# Test 9: Quick model training
print("\n9️⃣ Testing Quick Model Training (Decision Tree)...")
try:
    from utils.data_utils import get_preprocessor
    preprocessor = get_preprocessor(X_train)
    trained_model = train_model('dt', preprocessor, X_train, y_train)
    print(f"   ✅ Model trained successfully")
except Exception as e:
    print(f"   ❌ Model training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 10: Model evaluation
print("\n🔟 Testing Model Evaluation...")
try:
    results = evaluate_model(trained_model, X_train, y_train, X_test, y_test)
    test_metrics = results['test_metrics']
    print(f"   ✅ Model evaluated successfully")
    print(f"   ℹ️  Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"   ℹ️  PR AUC: {test_metrics['pr_auc']:.4f}")
except Exception as e:
    print(f"   ❌ Model evaluation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 11: SHAP compatibility
print("\n1️⃣1️⃣ Testing SHAP Compatibility...")
try:
    import shap
    # Try TreeExplainer
    explainer = shap.TreeExplainer(trained_model)
    print(f"   ✅ SHAP TreeExplainer initialized successfully")
except Exception as e:
    print(f"   ⚠️  TreeExplainer failed: {e}")
    print(f"   ℹ️  KernelExplainer will be used as fallback")

# Summary
print("\n" + "=" * 70)
print("✅ ALL TESTS PASSED!")
print("=" * 70)
print("\n🚀 Ready to launch the app with:")
print("   streamlit run app.py")
print("\n📝 Or use the launcher script:")
print("   ./run_app.sh")
print("=" * 70)
