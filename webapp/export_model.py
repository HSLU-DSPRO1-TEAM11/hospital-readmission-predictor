"""
Export Model Script

This script exports the trained XGBoost model, scaler, and feature columns
from the training pipeline for use in the webapp.

Run this script once from the notebooks directory (required for relative paths):
    cd notebooks
    python ../webapp/export_model.py

Reasoning:
- The notebooks train models dynamically but don't export them in a portable format
- We need to save the model, scaler, and exact feature column order
- This ensures the webapp can preprocess new data identically to training
"""

import os
import sys
import json
import pickle
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.feature_engineering import feature_engineering

RANDOM_STATE = 42

def export_model():
    """
    Retrain the best model (XGBoost with tuned hyperparameters) and export
    all necessary artifacts for the webapp.
    """
    print("Loading and preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler = feature_engineering()

    print(f"Training set: {X_train.shape[0]} samples, {X_train.shape[1]} features")

    # Feature selection (same as in hyperparameter_tuning.ipynb)
    print("Performing feature selection...")
    groups = {}
    for col in X_train.columns:
        base = re.sub(r'_[^_]+$', '', col)
        groups.setdefault(base, []).append(col)

    selector_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    selector_model.fit(X_train, y_train)

    importances = pd.Series(selector_model.feature_importances_, index=X_train.columns)
    top_features = importances.nlargest(25).index

    final_features = set()
    for g, cols in groups.items():
        if any(c in top_features for c in cols):
            final_features.update(cols)

    X_train = X_train[list(final_features)]

    # Clean column names for XGBoost compatibility
    X_train.columns = [re.sub(r'[^A-Za-z0-9_]+', '_', col) for col in X_train.columns]

    print(f"Selected {len(X_train.columns)} features after grouping")

    # Train XGBoost with best hyperparameters (from tuning)
    print("Training XGBoost with best hyperparameters...")
    best_params = {
        'subsample': 0.7,
        'scale_pos_weight': 2,
        'n_estimators': 200,
        'min_child_weight': 3,
        'max_depth': 3,
        'learning_rate': 0.1,
        'gamma': 0,
        'colsample_bytree': 0.9,
        'random_state': RANDOM_STATE,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)

    # Create model directory if it doesn't exist
    model_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_dir, 'best_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to: {model_path}")

    # Save scaler
    scaler_path = os.path.join(model_dir, 'scaler.pkl')
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Scaler saved to: {scaler_path}")

    # Save feature columns (order matters!)
    feature_columns = list(X_train.columns)
    columns_path = os.path.join(model_dir, 'feature_columns.json')
    with open(columns_path, 'w') as f:
        json.dump(feature_columns, f, indent=2)
    print(f"Feature columns saved to: {columns_path}")

    print(f"\nExport complete! {len(feature_columns)} features exported.")
    print("You can now run the webapp with: streamlit run webapp/app.py")

if __name__ == '__main__':
    export_model()
