#!/usr/bin/env python3
"""
Generate all visualization plots for model validation.

This script creates ROC curves, PR curves, confusion matrices,
KS plots, PSI charts, and other visualizations.
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_curve, 
    auc, 
    precision_recall_curve,
    confusion_matrix,
    calibration_curve
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def plot_roc_curve(y_true, y_pred_proba, output_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved ROC curve")


def plot_precision_recall_curve(y_true, y_pred_proba, output_path):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    plt.figure(figsize=(8, 8))
    plt.plot(recall, precision, color='blue', lw=2,
             label=f'PR curve (AUC = {pr_auc:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Precision-Recall curve")


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Good', 'Default'],
                yticklabels=['Good', 'Default'])
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved Confusion Matrix")


def plot_ks_statistic(y_true, y_pred_proba, output_path):
    """Plot KS statistic."""
    # Separate scores by class
    scores_0 = y_pred_proba[y_true == 0]
    scores_1 = y_pred_proba[y_true == 1]
    
    # Calculate CDFs
    x = np.linspace(0, 1, 100)
    cdf_0 = np.array([np.mean(scores_0 <= xi) for xi in x])
    cdf_1 = np.array([np.mean(scores_1 <= xi) for xi in x])
    
    # KS statistic
    ks = np.max(np.abs(cdf_1 - cdf_0))
    ks_idx = np.argmax(np.abs(cdf_1 - cdf_0))
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, cdf_0, label='Good (Class 0)', lw=2)
    plt.plot(x, cdf_1, label='Default (Class 1)', lw=2)
    plt.axvline(x[ks_idx], color='red', linestyle='--', 
                label=f'KS = {ks:.4f}')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Cumulative Distribution')
    plt.title('Kolmogorov-Smirnov Statistic')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'ks_statistic.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved KS statistic plot")


def plot_score_distribution(y_true, y_pred_proba, output_path):
    """Plot score distribution for good vs bad."""
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred_proba[y_true == 0], bins=50, alpha=0.5, 
             label='Good', color='green', density=True)
    plt.hist(y_pred_proba[y_true == 1], bins=50, alpha=0.5, 
             label='Default', color='red', density=True)
    plt.xlabel('Predicted Probability')
    plt.ylabel('Density')
    plt.title('Score Distribution: Good vs Default')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'score_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved score distribution")


def plot_calibration_curve(y_true, y_pred_proba, output_path):
    """Plot calibration curve."""
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_pred_proba, n_bins=10
    )
    
    plt.figure(figsize=(8, 8))
    plt.plot(mean_predicted_value, fraction_of_positives, 
             marker='o', lw=2, label='Model')
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, 
             color='gray', label='Perfectly calibrated')
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title('Calibration Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path / 'calibration_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved calibration curve")


def generate_all_plots(model_path, data_path, target_col, output_dir, threshold=0.5):
    """Generate all visualization plots."""
    print("Loading model and data...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    data = pd.read_csv(data_path)
    X = data.drop(columns=[target_col])
    y_true = data[target_col]
    
    # Get predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
    
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\nGenerating visualizations...")
    
    # Generate all plots
    plot_roc_curve(y_true, y_pred_proba, output_path)
    plot_precision_recall_curve(y_true, y_pred_proba, output_path)
    plot_confusion_matrix(y_true, y_pred, output_path)
    plot_ks_statistic(y_true, y_pred_proba, output_path)
    plot_score_distribution(y_true, y_pred_proba, output_path)
    plot_calibration_curve(y_true, y_pred_proba, output_path)
    
    print(f"\n✅ All visualizations saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate validation visualizations"
    )
    parser.add_argument("--model", required=True, help="Path to pickle model")
    parser.add_argument("--data", required=True, help="Path to validation data CSV")
    parser.add_argument("--target-col", default="target", help="Target column name")
    parser.add_argument("--output", default="results/plots/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    try:
        generate_all_plots(
            args.model,
            args.data,
            args.target_col,
            args.output,
            args.threshold
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())