#!/usr/bin/env python3
"""
Script for calculating model performance metrics.

This script calculates all key metrics for credit risk model validation including:
AUC, Gini, KS, Recall, Precision, F1, Accuracy, Specificity, and Confusion Matrix.
"""

import pickle
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    roc_curve
)


def calculate_gini(y_true, y_pred_proba):
    """Calculate Gini coefficient."""
    auc = roc_auc_score(y_true, y_pred_proba)
    return 2 * auc - 1


def calculate_ks_statistic(y_true, y_pred_proba):
    """Calculate Kolmogorov-Smirnov statistic."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    ks = max(tpr - fpr)
    return ks


def calculate_metrics(model_path, data_path, target_col, output_dir, threshold=0.5):
    """
    Calculate all metrics for model validation.
    
    Args:
        model_path: Path to pickle model
        data_path: Path to validation data
        target_col: Name of target column
        output_dir: Output directory for results
        threshold: Classification threshold
    """
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    print(f"Loading data from {data_path}...")
    data = pd.read_csv(data_path)
    
    # Separate features and target
    X = data.drop(columns=[target_col])
    y_true = data[target_col]
    
    print("Making predictions...")
    # Get probability predictions
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
    
    # Get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    
    # ROC AUC
    metrics['auc'] = float(roc_auc_score(y_true, y_pred_proba))
    
    # Gini
    metrics['gini'] = float(calculate_gini(y_true, y_pred_proba))
    
    # KS Statistic
    metrics['ks_statistic'] = float(calculate_ks_statistic(y_true, y_pred_proba))
    
    # Confusion Matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics['confusion_matrix'] = {
        'true_negative': int(tn),
        'false_positive': int(fp),
        'false_negative': int(fn),
        'true_positive': int(tp)
    }
    
    # Precision, Recall, F1
    metrics['precision'] = float(precision_score(y_true, y_pred))
    metrics['recall'] = float(recall_score(y_true, y_pred))
    metrics['f1_score'] = float(f1_score(y_true, y_pred))
    
    # Accuracy
    metrics['accuracy'] = float(accuracy_score(y_true, y_pred))
    
    # Specificity (TNR)
    metrics['specificity'] = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    # Sensitivity (same as Recall/TPR)
    metrics['sensitivity'] = metrics['recall']
    
    # Additional info
    metrics['threshold'] = threshold
    metrics['total_samples'] = len(y_true)
    metrics['positive_samples'] = int(y_true.sum())
    metrics['negative_samples'] = int(len(y_true) - y_true.sum())
    
    # Save metrics
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metrics_file = output_path / 'metrics.json'
    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Metrics saved to: {metrics_file}")
    print(f"\n📊 Key Metrics:")
    print(f"  AUC: {metrics['auc']:.4f}")
    print(f"  Gini: {metrics['gini']:.4f}")
    print(f"  KS: {metrics['ks_statistic']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1_score']:.4f}")
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Calculate model performance metrics"
    )
    parser.add_argument("--model", required=True, help="Path to pickle model")
    parser.add_argument("--data", required=True, help="Path to validation data CSV")
    parser.add_argument("--target-col", default="target", help="Target column name")
    parser.add_argument("--output", default="results/metrics/", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.5, help="Classification threshold")
    
    args = parser.parse_args()
    
    try:
        calculate_metrics(
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