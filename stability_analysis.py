#!/usr/bin/env python3
"""
PSI (Population Stability Index) and CSI (Characteristic Stability Index) analysis.

This script calculates stability metrics to assess model drift.
"""

import pandas as pd
import numpy as np
import json
import argparse
from pathlib import Path
import pickle
import warnings


def calculate_psi(expected, actual, buckets=10):
    """
    Calculate PSI (Population Stability Index).
    
    Args:
        expected: Expected distribution (baseline/training data)
        actual: Actual distribution (validation data)
        buckets: Number of buckets for binning
        
    Returns:
        PSI value
    """
    # Create bins based on expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    # Count observations in each bucket
    expected_counts = pd.cut(expected, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts()
    actual_counts = pd.cut(actual, bins=breakpoints, include_lowest=True, duplicates='drop').value_counts()
    
    # Calculate percentages
    expected_pct = expected_counts / len(expected)
    actual_pct = actual_counts / len(actual)
    
    # Avoid division by zero
    expected_pct = expected_pct.replace(0, 0.0001)
    actual_pct = actual_pct.replace(0, 0.0001)
    
    # Calculate PSI
    psi_values = (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
    psi = psi_values.sum()
    
    return psi


def analyze_stability(reference_data_path, validation_data_path, model_path, output_dir):
    """
    Perform stability analysis on features and model scores.
    
    Args:
        reference_data_path: Path to reference/training data
        validation_data_path: Path to validation data
        model_path: Path to model (for scoring)
        output_dir: Output directory
    """
    print("Loading data...")
    ref_data = pd.read_csv(reference_data_path)
    val_data = pd.read_csv(validation_data_path)
    
    print("Loading model...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    # Find common columns (excluding target if present)
    common_cols = list(set(ref_data.columns) & set(val_data.columns))
    
    # Calculate PSI for each feature
    psi_results = {}
    
    print("Calculating PSI for features...")
    for col in common_cols:
        # Skip non-numeric columns
        if not pd.api.types.is_numeric_dtype(ref_data[col]):
            continue
            
        try:
            psi = calculate_psi(ref_data[col].dropna(), val_data[col].dropna())
            psi_results[col] = {
                'psi': float(psi),
                'status': 'Stable' if psi < 0.1 else ('Moderate' if psi < 0.25 else 'Unstable')
            }
        except Exception as e:
            warnings.warn(f"Could not calculate PSI for {col}: {e}")
    
    # Calculate CSI for model scores
    print("Calculating CSI for model scores...")
    
    # Prepare features (exclude target column if present)
    target_cols = ['target', 'default_flag', 'default', 'bad', 'y']
    ref_features = ref_data.drop(columns=[c for c in target_cols if c in ref_data.columns], errors='ignore')
    val_features = val_data.drop(columns=[c for c in target_cols if c in val_data.columns], errors='ignore')
    
    # Get common features only
    common_features = list(set(ref_features.columns) & set(val_features.columns))
    
    try:
        if hasattr(model, 'predict_proba'):
            ref_scores = model.predict_proba(ref_features[common_features])[:, 1]
            val_scores = model.predict_proba(val_features[common_features])[:, 1]
        else:
            ref_scores = model.predict(ref_features[common_features])
            val_scores = model.predict(val_features[common_features])
        
        csi = calculate_psi(ref_scores, val_scores)
        
        csi_result = {
            'csi': float(csi),
            'status': 'Stable' if csi < 0.1 else ('Moderate' if csi < 0.25 else 'Unstable')
        }
    except Exception as e:
        warnings.warn(f"Could not calculate CSI: {e}")
        csi_result = {'csi': None, 'status': 'Error', 'error': str(e)}
    
    # Combine results
    results = {
        'feature_psi': psi_results,
        'model_csi': csi_result,
        'summary': {
            'total_features': len(psi_results),
            'stable_features': sum(1 for v in psi_results.values() if v['status'] == 'Stable'),
            'moderate_features': sum(1 for v in psi_results.values() if v['status'] == 'Moderate'),
            'unstable_features': sum(1 for v in psi_results.values() if v['status'] == 'Unstable')
        }
    }
    
    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results_file = output_path / 'stability_analysis.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Stability analysis saved to: {results_file}")
    print(f"\n📊 Summary:")
    print(f"  Total features analyzed: {results['summary']['total_features']}")
    print(f"  Stable (PSI < 0.1): {results['summary']['stable_features']}")
    print(f"  Moderate (0.1 ≤ PSI < 0.25): {results['summary']['moderate_features']}")
    print(f"  Unstable (PSI ≥ 0.25): {results['summary']['unstable_features']}")
    print(f"  Model CSI: {csi_result.get('csi', 'N/A')} ({csi_result['status']})")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Perform PSI/CSI stability analysis"
    )
    parser.add_argument("--reference-data", required=True, help="Path to reference/training data CSV")
    parser.add_argument("--validation-data", required=True, help="Path to validation data CSV")
    parser.add_argument("--model", required=True, help="Path to pickle model")
    parser.add_argument("--output", default="results/stability/", help="Output directory")
    
    args = parser.parse_args()
    
    try:
        analyze_stability(
            args.reference_data,
            args.validation_data,
            args.model,
            args.output
        )
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())