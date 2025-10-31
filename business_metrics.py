#!/usr/bin/env python3
"""
Calculate business metrics for credit risk models.

This script calculates approval rates, bad rates, and expected losses
across different threshold values.
"""

import pickle
import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path


def calculate_business_metrics(model_path, data_path, target_col, output_dir, 
                               cost_fn=1.0, cost_fp=0.1):
    """
    Calculate business metrics at different thresholds.
    
    Args:
        model_path: Path to pickle model
        data_path: Path to validation data
        target_col: Name of target column
        output_dir: Output directory
        cost_fn: Cost of False Negative (missed default)
        cost_fp: Cost of False Positive (rejected good customer)
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
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X)[:, 1]
    else:
        y_pred_proba = model.predict(X)
    
    # Test different thresholds
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    print("Calculating metrics for different thresholds...")
    for threshold in thresholds:
        # Predictions at this threshold (1 = default, 0 = good)
        # We REJECT if predicted probability >= threshold
        approved = (y_pred_proba < threshold)
        
        # Calculate metrics
        total = len(y_true)
        approved_count = approved.sum()
        approval_rate = approved_count / total
        
        # Among approved, how many defaulted?
        approved_defaults = (y_true[approved] == 1).sum()
        bad_rate = approved_defaults / approved_count if approved_count > 0 else 0
        
        # Calculate costs
        # FN: We approved (pred=0) but they defaulted (actual=1)
        false_negatives = ((y_pred_proba < threshold) & (y_true == 1)).sum()
        
        # FP: We rejected (pred=1) but they were good (actual=0)
        false_positives = ((y_pred_proba >= threshold) & (y_true == 0)).sum()
        
        total_cost = (false_negatives * cost_fn) + (false_positives * cost_fp)
        avg_cost_per_application = total_cost / total
        
        expected_loss = bad_rate * approved_count  # Number of expected defaults
        
        results.append({
            'threshold': float(threshold),
            'approval_rate': float(approval_rate),
            'approved_count': int(approved_count),
            'bad_rate': float(bad_rate),
            'approved_defaults': int(approved_defaults),
            'false_negatives': int(false_negatives),
            'false_positives': int(false_positives),
            'total_cost': float(total_cost),
            'avg_cost_per_application': float(avg_cost_per_application),
            'expected_loss': float(expected_loss)
        })
    
    # Find optimal threshold (minimize cost)
    optimal_idx = min(range(len(results)), key=lambda i: results[i]['total_cost'])
    optimal_threshold = results[optimal_idx]
    
    output_data = {
        'threshold_analysis': results,
        'optimal_threshold': optimal_threshold,
        'cost_parameters': {
            'cost_fn': cost_fn,
            'cost_fp': cost_fp
        }
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'business_metrics.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Business metrics saved to: {results_file}")
    print(f"\n📊 Optimal Threshold: {optimal_threshold['threshold']:.2f}")
    print(f"  Approval Rate: {optimal_threshold['approval_rate']:.2%}")
    print(f"  Bad Rate: {optimal_threshold['bad_rate']:.2%}")
    print(f"  Total Cost: {optimal_threshold['total_cost']:.2f}")
    
    return output_data


def main():
    parser = argparse.ArgumentParser(
        description="Calculate business metrics"
    )
    parser.add_argument("--model", required=True, help="Path to pickle model")
    parser.add_argument("--data", required=True, help="Path to validation data CSV")
    parser.add_argument("--target-col", default="target", help="Target column name")
    parser.add_argument("--output", default="results/business/", help="Output directory")
    parser.add_argument("--cost-fn", type=float, default=1.0, help="Cost of False Negative")
    parser.add_argument("--cost-fp", type=float, default=0.1, help="Cost of False Positive")
    
    args = parser.parse_args()
    
    try:
        calculate_business_metrics(
            args.model,
            args.data,
            args.target_col,
            args.output,
            args.cost_fn,
            args.cost_fp
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())