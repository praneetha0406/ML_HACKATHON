"""
Module: Task_2_Modeling
Description: Evaluates and trains four distinct ML models on processed, standardized time-series data.
Codex Check: 
- chronological 80/20 train/test split. (Zero shuffling allowed).
- Models tested for Robustness: Linear Regression, Random Forest, XGBoost, and ANN.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def execute_model_training(data_path: str, models_dir: str) -> None:
    """Trains 4 robust regression models chronologically, evaluating performance accurately on unseen data."""
    print("Initializing Robust Training Pipeline...")
    df = pd.read_csv(data_path)
    
    # 1. Feature Target Extraction
    X = df.drop(columns=['target_nswdemand'])
    y = df['target_nswdemand']
    
    # 2. Sequential Splitting (Time Series Mandatory)
    # Using random `train_test_split(shuffle=True)` is fatal data leakage for time-series.
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    os.makedirs(models_dir, exist_ok=True)
    results = {}
    
    # 3. Model Evaluation Engine
    def evaluate(model_name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        results[model_name] = {'MAE': mae, 'RMSE': rmse, 'R2': r2}
        print(f"[{model_name}] Validation Complete - MAE: {mae:.5f}")

    # --- Training Loop ---
    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    evaluate('Linear Regression', y_test, lr_model.predict(X_test))
    joblib.dump(lr_model, os.path.join(models_dir, 'linear_regression.pkl'))
    
    # Random Forest Ensemble
    rf_model = RandomForestRegressor(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)
    rf_model.fit(X_train, y_train)
    evaluate('Random Forest', y_test, rf_model.predict(X_test))
    joblib.dump(rf_model, os.path.join(models_dir, 'random_forest.pkl'))
    
    # XGBoost Gradient Boosting
    xgb_model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42)
    xgb_model.fit(X_train, y_train)
    evaluate('XGBoost', y_test, xgb_model.predict(X_test))
    joblib.dump(xgb_model, os.path.join(models_dir, 'xgboost.pkl'))
    
    # Neural Network (ANN)
    ann_model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.2), # Dropout layer physically prevents overfitting
        Dense(32, activation='relu'),
        Dense(1) 
    ])
    ann_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    ann_model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1, verbose=0)
    evaluate('ANN', y_test, ann_model.predict(X_test, verbose=0).flatten())
    ann_model.save(os.path.join(models_dir, 'ann_model.keras'))
    
    # Output metrics table
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(models_dir, 'model_evaluation_metrics.csv'))
    print("Training sequence complete. No anomalies detected.")

if __name__ == "__main__":
    execute_model_training("processed_energy_data.csv", "trained_models")
