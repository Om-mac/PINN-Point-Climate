"""
Test boundary value analysis for flood prediction model.
Tests different precipitation scenarios to verify model behavior.
"""
import pickle
import pandas as pd
import numpy as np

# Load the trained model
print("Loading model...")
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

print(f"Model loaded with {len(feature_columns)} features")
print(f"Features: {feature_columns}\n")

# Load a sample station for realistic catchment characteristics
stations_df = pd.read_csv('DATA/catchment_characteristics_indofloods.csv')
metadata_df = pd.read_csv('DATA/metadata_indofloods.csv')

# Merge to get a sample station
sample = stations_df.merge(metadata_df, on='GaugeID').iloc[0]

print("=" * 60)
print("BOUNDARY VALUE ANALYSIS - PRECIPITATION VARIATIONS")
print("=" * 60)

# Test different precipitation scenarios
scenarios = [
    {
        'name': '1. Zero Precipitation (0mm all days)',
        'precip': [0] * 10
    },
    {
        'name': '2. Very Light Rain (5mm all days)',
        'precip': [5] * 10
    },
    {
        'name': '3. Light Rain (15mm all days)',
        'precip': [15] * 10
    },
    {
        'name': '4. Moderate Rain (30mm all days)',
        'precip': [30] * 10
    },
    {
        'name': '5. Heavy Rain (60mm all days)',
        'precip': [60] * 10
    },
    {
        'name': '6. Very Heavy Rain (100mm all days)',
        'precip': [100] * 10
    },
    {
        'name': '7. Extreme Rain (150mm all days)',
        'precip': [150] * 10
    },
    {
        'name': '8. Recent Heavy Rain (100mm yesterday, then decreasing)',
        'precip': [100, 80, 60, 40, 20, 10, 5, 0, 0, 0]
    },
    {
        'name': '9. Building Rain Pattern (increasing)',
        'precip': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    }
]

for scenario in scenarios:
    precip = scenario['precip']
    
    # Build feature dictionary
    features = {}
    
    # Precipitation features
    for i, val in enumerate(precip, 1):
        features[f'T{i}d'] = val
    
    # API features
    features['API_3day'] = sum(precip[:3])
    features['API_7day'] = sum(precip[:7])
    features['API_10day'] = sum(precip[:10])
    
    # Catchment characteristics from sample station
    features['Drainage Area'] = sample.get('Drainage Area', 0)
    features['Catchment Relief'] = sample.get('Catchment Relief', 0)
    features['Catchment Length'] = sample.get('Catchment Length', 0)
    features['Relief Ratio'] = sample.get('Relief Ratio', 0)
    features['Drainage Density'] = sample.get('Drainage Density', 0)
    features['Stream Order'] = sample.get('Stream Order', 0)
    features['Catchment Area'] = sample.get('Catchment Area', 0)
    
    # Encoded categorical features (use defaults)
    for col in feature_columns:
        if col.endswith('_encoded') and col not in features:
            features[col] = 0
    
    # Ensure all features are present in correct order
    feature_vector = [features.get(col, 0) for col in feature_columns]
    
    # Make prediction
    X = pd.DataFrame([feature_vector], columns=feature_columns)
    probability = model.predict_proba(X)[0][1]  # Probability of flood (class 1)
    
    print(f"\n{scenario['name']}")
    print(f"  API_3day: {features['API_3day']:.1f}mm")
    print(f"  API_7day: {features['API_7day']:.1f}mm")
    print(f"  API_10day: {features['API_10day']:.1f}mm")
    print(f"  🎯 Flood Probability: {probability*100:.1f}%")
    
    if probability >= 0.7:
        print(f"  ⚠️  CRITICAL RISK - PINN Stage 2 Triggered")
    elif probability >= 0.4:
        print(f"  ⚡ HIGH RISK")
    elif probability >= 0.2:
        print(f"  ⚠️  MODERATE RISK")
    else:
        print(f"  ✅ LOW RISK")

print("\n" + "=" * 60)
print("EXPECTED BEHAVIOR:")
print("=" * 60)
print("✓ 0mm precipitation → Should be <10% (LOW RISK)")
print("✓ Light rain (5-15mm) → Should be 10-30% (LOW-MODERATE RISK)")
print("✓ Heavy rain (60-100mm) → Should be >70% (CRITICAL RISK)")
print("✓ Extreme rain (>150mm) → Should be >90% (CRITICAL RISK)")
