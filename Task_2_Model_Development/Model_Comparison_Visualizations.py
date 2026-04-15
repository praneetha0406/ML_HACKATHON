"""
Module: Model Performance Visualizations
Description: Generates comprehensive, publication-quality comparison charts for all
trained models. Covers metric comparisons, prediction accuracy, residual analysis,
and learning curves.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import tensorflow as tf
import os
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'figure.facecolor': '#0f0f1a',
    'axes.facecolor': '#1a1a2e',
    'axes.edgecolor': '#444466',
    'axes.labelcolor': '#ccccff',
    'xtick.color': '#aaaacc',
    'ytick.color': '#aaaacc',
    'text.color': '#ccccff',
    'grid.color': '#2a2a4a',
    'grid.alpha': 0.5,
    'font.family': 'DejaVu Sans',
})

COLORS = ['#7c83fd', '#96fbc4', '#f7971e', '#fd7c7c']
MODEL_NAMES = ['Linear Regression', 'Random Forest', 'XGBoost', 'ANN']
OUTPUT_DIR = 'model_comparison_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_all():
    df = pd.read_csv('processed_energy_data.csv')
    X = df.drop(columns=['target_nswdemand'])
    y = df['target_nswdemand']
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    lr  = joblib.load('trained_models/linear_regression.pkl')
    rf  = joblib.load('trained_models/random_forest.pkl')
    xgb = joblib.load('trained_models/xgboost.pkl')
    ann = tf.keras.models.load_model('trained_models/ann_model.keras')
    preds_test = {
        'Linear Regression': lr.predict(X_test),
        'Random Forest':     rf.predict(X_test),
        'XGBoost':           xgb.predict(X_test),
        'ANN':               ann.predict(X_test, verbose=0).flatten(),
    }
    preds_train = {
        'Linear Regression': lr.predict(X_train),
        'Random Forest':     rf.predict(X_train),
        'XGBoost':           xgb.predict(X_train),
        'ANN':               ann.predict(X_train, verbose=0).flatten(),
    }
    return y_train.values, y_test.values, preds_train, preds_test


def plot_metric_bars(metrics_csv):
    df = pd.read_csv(metrics_csv, index_col=0)
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Metric Comparison', fontsize=18, fontweight='bold', color='#ccccff', y=1.02)
    for idx, (metric, ax) in enumerate(zip(['MAE', 'RMSE', 'R2'], axes)):
        vals = df[metric]
        bars = ax.bar(vals.index, vals.values, color=COLORS, edgecolor='#ffffff22', linewidth=0.5, width=0.5)
        ax.set_title(metric, fontsize=14, fontweight='bold', color=COLORS[idx])
        ax.set_ylabel(metric, fontsize=11)
        ax.tick_params(axis='x', rotation=15)
        ax.grid(axis='y', alpha=0.4)
        for bar, val in zip(bars, vals.values):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0005,
                    f'{val:.5f}', ha='center', va='bottom', fontsize=9, color='white')
        best_idx = vals.idxmin() if metric != 'R2' else vals.idxmax()
        best_pos = list(vals.index).index(best_idx)
        bars[best_pos].set_edgecolor('#ffffff')
        bars[best_pos].set_linewidth(2)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '01_metric_comparison_bars.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_radar(metrics_csv):
    df = pd.read_csv(metrics_csv, index_col=0)
    df_norm = df.copy()
    df_norm['MAE']  = 1 - (df['MAE']  / df['MAE'].max())
    df_norm['RMSE'] = 1 - (df['RMSE'] / df['RMSE'].max())
    df_norm['R2']   = df['R2']
    categories = ['MAE\n(inverted)', 'RMSE\n(inverted)', 'R2 Score']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    for i, (model, row) in enumerate(df_norm.iterrows()):
        vals = row[['MAE', 'RMSE', 'R2']].tolist()
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, label=model, color=COLORS[i])
        ax.fill(angles, vals, alpha=0.12, color=COLORS[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, color='#ccccff', fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.50', '0.75', '1.0'], color='#888899', fontsize=8)
    ax.yaxis.grid(True, color='#333355', linewidth=0.5)
    ax.xaxis.grid(True, color='#333355', linewidth=0.5)
    ax.set_title('Model Radar Comparison\n(Higher = Better for All Axes)',
                 color='#ccccff', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1),
              facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccff')
    path = os.path.join(OUTPUT_DIR, '02_radar_comparison.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_actual_vs_predicted(y_test, preds_test):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Actual vs Predicted Energy Demand — All Models',
                 fontsize=16, fontweight='bold', color='#ccccff')
    axes = axes.flatten()
    sample_idx = np.linspace(0, len(y_test)-1, 500, dtype=int)
    y_sample = y_test[sample_idx]
    for i, (model_name, preds) in enumerate(preds_test.items()):
        ax = axes[i]
        pred_sample = preds[sample_idx]
        ax.scatter(y_sample, pred_sample, alpha=0.5, s=12, color=COLORS[i], label='Predictions')
        lims = [min(y_sample.min(), pred_sample.min()), max(y_sample.max(), pred_sample.max())]
        ax.plot(lims, lims, 'w--', linewidth=1.5, label='Perfect Fit', alpha=0.7)
        ax.set_xlabel('Actual Demand', fontsize=10)
        ax.set_ylabel('Predicted Demand', fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight='bold', color=COLORS[i])
        ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccff')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '03_actual_vs_predicted.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_residuals(y_test, preds_test):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Residual Error Distribution — All Models',
                 fontsize=16, fontweight='bold', color='#ccccff')
    axes = axes.flatten()
    for i, (model_name, preds) in enumerate(preds_test.items()):
        residuals = y_test - preds
        ax = axes[i]
        ax.hist(residuals, bins=60, color=COLORS[i], alpha=0.8, edgecolor='none')
        ax.axvline(0, color='white', linewidth=1.5, linestyle='--', alpha=0.8, label='Zero Error')
        ax.axvline(residuals.mean(), color='#ffcc00', linewidth=1.5, linestyle=':', label=f'Mean: {residuals.mean():.5f}')
        ax.set_xlabel('Residual (Actual - Predicted)', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title(model_name, fontsize=12, fontweight='bold', color=COLORS[i])
        ax.legend(fontsize=9, facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccff')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '04_residual_distributions.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_train_vs_test_mae(y_train, y_test, preds_train, preds_test):
    from sklearn.metrics import mean_absolute_error
    train_maes = [mean_absolute_error(y_train, preds_train[m]) for m in MODEL_NAMES]
    test_maes  = [mean_absolute_error(y_test,  preds_test[m])  for m in MODEL_NAMES]
    x = np.arange(len(MODEL_NAMES))
    width = 0.35
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    ax.bar(x - width/2, train_maes, width, label='Train MAE', color=COLORS, alpha=0.6, edgecolor='white', linewidth=0.5)
    bars_test = ax.bar(x + width/2, test_maes, width, label='Test MAE',  color=COLORS, alpha=1.0, edgecolor='white', linewidth=0.8)
    for i, (tr, te) in enumerate(zip(train_maes, test_maes)):
        ax.text(x[i]-width/2, tr+0.0001, f'{tr:.5f}', ha='center', va='bottom', fontsize=8, color='#aaaacc')
        ax.text(x[i]+width/2, te+0.0001, f'{te:.5f}', ha='center', va='bottom', fontsize=8, color='white')
    ax.set_xticks(x); ax.set_xticklabels(MODEL_NAMES, fontsize=11)
    ax.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax.set_title('Overfitting Analysis: Train MAE vs Test MAE\n(Closer bars = More Robust and Generalized Model)',
                 fontsize=14, fontweight='bold', color='#ccccff')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccff', fontsize=11)
    ax.grid(axis='y', alpha=0.4)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '05_train_vs_test_mae.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_forecast_overlay(y_test, preds_test):
    fig, ax = plt.subplots(figsize=(18, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    n = min(500, len(y_test))
    x_axis = np.arange(n)
    ax.plot(x_axis, y_test[-n:], color='white', linewidth=1.5, label='Actual Demand', alpha=0.9, zorder=5)
    for i, (model_name, preds) in enumerate(preds_test.items()):
        ax.plot(x_axis, preds[-n:], color=COLORS[i], linewidth=1,
                label=model_name, alpha=0.7, linestyle='--' if i > 1 else '-')
    ax.set_xlabel('Time Step (Half-Hour Periods)', fontsize=12)
    ax.set_ylabel('Scaled Energy Demand', fontsize=12)
    ax.set_title('Time-Series Forecast Overlay — All Models vs Actual\n(Final 500 Test Periods)',
                 fontsize=14, fontweight='bold', color='#ccccff')
    ax.legend(facecolor='#1a1a2e', edgecolor='#444466', labelcolor='#ccccff', fontsize=10, ncol=5)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '06_forecast_overlay.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


def plot_r2_leaderboard(metrics_csv):
    df = pd.read_csv(metrics_csv, index_col=0).sort_values('R2', ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#0f0f1a')
    ax.set_facecolor('#1a1a2e')
    bars = ax.barh(df.index, df['R2'] * 100, color=COLORS[::-1], edgecolor='#ffffff22', height=0.5)
    for bar, val in zip(bars, df['R2'] * 100):
        ax.text(bar.get_width() - 0.3, bar.get_y() + bar.get_height()/2,
                f'{val:.2f}%', va='center', ha='right', fontsize=11, fontweight='bold', color='white')
    ax.set_xlim(95, 100.5)
    ax.set_xlabel('R2 Score (%)', fontsize=12)
    ax.set_title('R2 Leaderboard — Model Accuracy Ranking\n(How well the model explains energy variance)',
                 fontsize=14, fontweight='bold', color='#ccccff')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, '07_r2_leaderboard.png')
    fig.savefig(path, bbox_inches='tight', dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    print(f"Saved: {path}")


if __name__ == "__main__":
    print("Loading data and models...")
    y_train, y_test, preds_train, preds_test = load_all()
    metrics_csv = 'trained_models/model_evaluation_metrics.csv'
    print("\nGenerating all 7 comparison charts...\n")
    plot_metric_bars(metrics_csv)
    plot_radar(metrics_csv)
    plot_actual_vs_predicted(y_test, preds_test)
    plot_residuals(y_test, preds_test)
    plot_train_vs_test_mae(y_train, y_test, preds_train, preds_test)
    plot_forecast_overlay(y_test, preds_test)
    plot_r2_leaderboard(metrics_csv)
    print(f"\nAll charts saved to: {OUTPUT_DIR}/")
