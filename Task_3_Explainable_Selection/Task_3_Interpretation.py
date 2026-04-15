import pandas as pd
import numpy as np
import os
import joblib
import xgboost as xgb
import tensorflow as tf
import shap
import matplotlib.pyplot as plt

def select_and_explain(data_path, models_dir, output_dir):
    print("Loading model metrics and auto-selecting best model...")
    metrics_file = os.path.join(models_dir, 'model_evaluation_metrics.csv')
    if not os.path.exists(metrics_file):
        raise FileNotFoundError(f"{metrics_file} not found. Ensure Task 2 finished successfully.")
    
    metrics_df = pd.read_csv(metrics_file, index_col=0)
    print("Evaluation Metrics:")
    print(metrics_df)
    
    # Select Best Model based on Lowest MAE
    best_model_name = metrics_df['MAE'].idxmin()
    print(f"\n[Auto Model Selection] -> Best performing model is {best_model_name} with MAE {metrics_df.loc[best_model_name, 'MAE']:.4f}")
    
    # Load the best model
    if best_model_name == 'Linear Regression':
        model_path = os.path.join(models_dir, 'linear_regression.pkl')
        model = joblib.load(model_path)
    elif best_model_name == 'Random Forest':
        model_path = os.path.join(models_dir, 'random_forest.pkl')
        model = joblib.load(model_path)
    elif best_model_name == 'XGBoost':
        model_path = os.path.join(models_dir, 'xgboost.pkl')
        model = joblib.load(model_path)
    elif best_model_name == 'ANN':
        model_path = os.path.join(models_dir, 'ann_model.keras')
        model = tf.keras.models.load_model(model_path)
    else:
        raise ValueError(f"Unknown best model {best_model_name}")

    print("Loading test data for SHAP interpretation...")
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target_nswdemand'])
    split_idx = int(len(df) * 0.8)
    X_test = X.iloc[split_idx:]
    
    # Since background computation of SHAP values can be slow, use a random sample 
    # of 500 instances for global explanations
    X_sample = shap.sample(X_test, 500)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Calculating SHAP values for {best_model_name}...")
    
    # Choose appropriate SHAP explainer
    if best_model_name in ['Random Forest', 'XGBoost']:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    elif best_model_name == 'Linear Regression':
        explainer = shap.LinearExplainer(model, X_train) # Note: requires X_train context usually, ignoring for simplicity
        shap_values = explainer.shap_values(X_sample)
    elif best_model_name == 'ANN':
        explainer = shap.DeepExplainer(model, X_sample.values)
        shap_values = explainer.shap_values(X_sample.values)
        # DeepExplainer returns a list for regression sometimes, unwrap it
        if isinstance(shap_values, list):
            shap_values = shap_values[0]

    # Handle shape inconsistency for Random Forest (sometimes returns list of arrays for trees)
    if isinstance(shap_values, list) and len(shap_values) > 0 and isinstance(shap_values[0], np.ndarray):
            # For regressors it should be a single array, but just in case:
            pass 

    print("Generating SHAP Summary Plot (Global Explainability)...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.savefig(os.path.join(output_dir, f'shap_summary_{best_model_name.replace(" ", "_")}.png'), bbox_inches='tight')
    plt.close()
    
    print("Generating SHAP Bar Plot (Feature Importance)...")
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False)
    plt.savefig(os.path.join(output_dir, f'shap_bar_{best_model_name.replace(" ", "_")}.png'), bbox_inches='tight')
    plt.close()
    
    print(f"XAI Interpretations completely saved in {output_dir}")

if __name__ == "__main__":
    data_file = "processed_energy_data.csv"
    models_input_dir = "trained_models"
    plots_output_dir = "xai_plots"
    select_and_explain(data_file, models_input_dir, plots_output_dir)
