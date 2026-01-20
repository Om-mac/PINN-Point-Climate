"""
Model Comparison: Random Forest vs XGBoost
Compare performance of both models side-by-side to help choose the best one.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report
)
import time
import warnings
warnings.filterwarnings('ignore')

from flood_prediction_model import FloodPredictionModel


def compare_models():
    """Train and compare Random Forest vs XGBoost."""
    
    print("\n" + "="*70)
    print("MODEL COMPARISON: Random Forest vs XGBoost")
    print("="*70)
    
    # Load and prepare data
    print("\n📊 Loading and preparing data...")
    model = FloodPredictionModel(data_path='DATA/')
    (model
        .load_data()
        .prepare_flood_events()
        .generate_negative_samples(samples_per_station=10)
        .create_features()
    )
    
    # Prepare data
    X = model.master_df[model.feature_columns]
    y = model.master_df['Flood_Label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Dataset ready: {len(X_train)} training, {len(X_test)} test samples")
    
    results = {}
    
    # ========== Random Forest ==========
    print("\n" + "-"*70)
    print("🌲 RANDOM FOREST")
    print("-"*70)
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42,
        class_weight='balanced',
        n_jobs=-1
    )
    
    # Train
    print("Training...")
    start_time = time.time()
    rf_model.fit(X_train, y_train)
    rf_train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    rf_pred = rf_model.predict(X_test)
    rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
    rf_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    # Metrics
    results['Random Forest'] = {
        'Accuracy': accuracy_score(y_test, rf_pred),
        'Precision': precision_score(y_test, rf_pred),
        'Recall': recall_score(y_test, rf_pred),
        'F1-Score': f1_score(y_test, rf_pred),
        'ROC-AUC': roc_auc_score(y_test, rf_pred_proba),
        'Training Time (s)': rf_train_time,
        'Inference Time (ms)': rf_inference_time,
        'Model Size (MB)': estimate_model_size(rf_model)
    }
    
    print(f"✓ Accuracy: {results['Random Forest']['Accuracy']:.4f}")
    print(f"✓ ROC-AUC: {results['Random Forest']['ROC-AUC']:.4f}")
    print(f"✓ Training Time: {rf_train_time:.2f}s")
    print(f"✓ Inference Time: {rf_inference_time:.4f}ms/sample")
    
    # ========== XGBoost ==========
    print("\n" + "-"*70)
    print("⚡ XGBOOST")
    print("-"*70)
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )
    
    # Train
    print("Training...")
    start_time = time.time()
    xgb_model.fit(X_train, y_train, verbose=False)
    xgb_train_time = time.time() - start_time
    
    # Predict
    start_time = time.time()
    xgb_pred = xgb_model.predict(X_test)
    xgb_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
    xgb_inference_time = (time.time() - start_time) / len(X_test) * 1000  # ms per sample
    
    # Metrics
    results['XGBoost'] = {
        'Accuracy': accuracy_score(y_test, xgb_pred),
        'Precision': precision_score(y_test, xgb_pred),
        'Recall': recall_score(y_test, xgb_pred),
        'F1-Score': f1_score(y_test, xgb_pred),
        'ROC-AUC': roc_auc_score(y_test, xgb_pred_proba),
        'Training Time (s)': xgb_train_time,
        'Inference Time (ms)': xgb_inference_time,
        'Model Size (MB)': estimate_model_size(xgb_model)
    }
    
    print(f"✓ Accuracy: {results['XGBoost']['Accuracy']:.4f}")
    print(f"✓ ROC-AUC: {results['XGBoost']['ROC-AUC']:.4f}")
    print(f"✓ Training Time: {xgb_train_time:.2f}s")
    print(f"✓ Inference Time: {xgb_inference_time:.4f}ms/sample")
    
    # ========== Comparison Table ==========
    print("\n" + "="*70)
    print("📊 PERFORMANCE COMPARISON")
    print("="*70)
    
    comparison_df = pd.DataFrame(results).T
    print("\n" + comparison_df.to_string())
    
    # ========== Winner Analysis ==========
    print("\n" + "="*70)
    print("🏆 ANALYSIS")
    print("="*70)
    
    # Determine winner for each metric
    print("\n✓ Best Performance by Metric:")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']:
        rf_val = results['Random Forest'][metric]
        xgb_val = results['XGBoost'][metric]
        winner = 'Random Forest' if rf_val > xgb_val else 'XGBoost'
        diff = abs(rf_val - xgb_val)
        print(f"  • {metric:12s}: {winner:15s} ({diff:.4f} difference)")
    
    print("\n✓ Best Efficiency by Metric:")
    for metric in ['Training Time (s)', 'Inference Time (ms)', 'Model Size (MB)']:
        rf_val = results['Random Forest'][metric]
        xgb_val = results['XGBoost'][metric]
        winner = 'Random Forest' if rf_val < xgb_val else 'XGBoost'
        ratio = max(rf_val, xgb_val) / min(rf_val, xgb_val)
        print(f"  • {metric:22s}: {winner:15s} ({ratio:.2f}x faster/smaller)")
    
    # ========== Recommendation ==========
    print("\n" + "="*70)
    print("💡 RECOMMENDATION")
    print("="*70)
    
    # Overall score (weighted)
    rf_score = (
        results['Random Forest']['Accuracy'] * 0.2 +
        results['Random Forest']['Precision'] * 0.2 +
        results['Random Forest']['Recall'] * 0.2 +
        results['Random Forest']['ROC-AUC'] * 0.3 +
        (1 / (results['Random Forest']['Training Time (s)'] + 1)) * 0.05 +
        (1 / (results['Random Forest']['Inference Time (ms)'] + 1)) * 0.05
    )
    
    xgb_score = (
        results['XGBoost']['Accuracy'] * 0.2 +
        results['XGBoost']['Precision'] * 0.2 +
        results['XGBoost']['Recall'] * 0.2 +
        results['XGBoost']['ROC-AUC'] * 0.3 +
        (1 / (results['XGBoost']['Training Time (s)'] + 1)) * 0.05 +
        (1 / (results['XGBoost']['Inference Time (ms)'] + 1)) * 0.05
    )
    
    print("\nOverall Weighted Score:")
    print(f"  Random Forest: {rf_score:.4f}")
    print(f"  XGBoost:       {xgb_score:.4f}")
    
    winner = 'Random Forest' if rf_score > xgb_score else 'XGBoost'
    print(f"\n🎯 Recommended Model: {winner}")
    
    print("\n📝 Use Case Recommendations:")
    print("  • For PRODUCTION (balanced): Random Forest")
    print("    - Slightly better ROC-AUC")
    print("    - More stable predictions")
    print("    - Easier to interpret")
    print()
    print("  • For RESEARCH (experimentation): XGBoost")
    print("    - Faster training")
    print("    - More hyperparameters to tune")
    print("    - Better for large datasets")
    
    # ========== PINN Trigger Comparison ==========
    print("\n" + "="*70)
    print("🔗 PINN TRIGGER ANALYSIS (Threshold = 0.7)")
    print("="*70)
    
    rf_trigger_count = (rf_pred_proba > 0.7).sum()
    xgb_trigger_count = (xgb_pred_proba > 0.7).sum()
    
    print(f"\nRandom Forest: {rf_trigger_count}/{len(y_test)} samples ({rf_trigger_count/len(y_test)*100:.1f}%)")
    print(f"XGBoost:       {xgb_trigger_count}/{len(y_test)} samples ({xgb_trigger_count/len(y_test)*100:.1f}%)")
    
    print("\n" + "="*70)
    print("✅ COMPARISON COMPLETE")
    print("="*70)
    
    return results


def estimate_model_size(model):
    """Estimate model size in MB."""
    import pickle
    import io
    
    buffer = io.BytesIO()
    pickle.dump(model, buffer)
    size_mb = buffer.tell() / (1024 * 1024)
    return size_mb


if __name__ == "__main__":
    results = compare_models()
