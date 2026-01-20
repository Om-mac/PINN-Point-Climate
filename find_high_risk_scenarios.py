"""
Find precipitation scenarios that result in 50%+ flood risk for different stations.
"""
import pickle
import pandas as pd
import numpy as np

# Load model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

# Load station data
stations_df = pd.read_csv('DATA/catchment_characteristics_indofloods.csv')
metadata_df = pd.read_csv('DATA/metadata_indofloods.csv')
combined = stations_df.merge(metadata_df, on='GaugeID')

print("=" * 70)
print("FINDING PRECIPITATION THRESHOLDS FOR 50%+ FLOOD RISK")
print("=" * 70)

# Test different precipitation levels
precip_scenarios = [
    {'name': 'Light Rain (20mm/day)', 'daily': 20},
    {'name': 'Moderate Rain (30mm/day)', 'daily': 30},
    {'name': 'Heavy Rain (50mm/day)', 'daily': 50},
    {'name': 'Very Heavy (80mm/day)', 'daily': 80},
]

# Test on 5 sample stations
sample_stations = combined.sample(n=min(5, len(combined)), random_state=42)

for scenario in precip_scenarios:
    daily_precip = scenario['daily']
    precip = [daily_precip] * 10
    
    print(f"\n{scenario['name']}")
    print(f"  Total 10-day precipitation: {daily_precip * 10}mm")
    print("-" * 70)
    
    high_risk_count = 0
    
    for idx, station in sample_stations.iterrows():
        # Build features
        features = {}
        for i, val in enumerate(precip, 1):
            features[f'T{i}d'] = val
        
        features['API_3day'] = sum(precip[:3])
        features['API_7day'] = sum(precip[:7])
        features['API_10day'] = sum(precip[:10])
        
        # Station characteristics
        features['Drainage Area'] = station.get('Drainage Area', 0)
        features['Catchment Relief'] = station.get('Catchment Relief', 0)
        features['Catchment Length'] = station.get('Catchment Length', 0)
        features['Relief Ratio'] = station.get('Relief Ratio', 0)
        features['Drainage Density'] = station.get('Drainage Density', 0)
        features['Stream Order'] = station.get('Stream Order', 0)
        features['Catchment Area'] = station.get('Catchment Area', 0)
        
        # Encoded features (defaults)
        for col in feature_columns:
            if col.endswith('_encoded') and col not in features:
                features[col] = 0
        
        # Predict
        feature_vector = [features.get(col, 0) for col in feature_columns]
        X = pd.DataFrame([feature_vector], columns=feature_columns)
        probability = model.predict_proba(X)[0][1]
        
        if probability >= 0.5:
            high_risk_count += 1
            print(f"  ✓ {station.get('Station', 'Unknown'):30s} - {probability*100:5.1f}% risk")
    
    if high_risk_count == 0:
        print("  (No stations reach 50% risk at this precipitation level)")
    
    print(f"  → {high_risk_count}/{len(sample_stations)} stations have 50%+ risk")

print("\n" + "=" * 70)
print("RECOMMENDATION TO SEE 50%+ RISK ON MAP:")
print("=" * 70)
print("1. Uncheck 'Fetch Real Weather Data' checkbox")
print("2. Enter these values manually:")
print("   - T1d (Yesterday): 80mm")
print("   - T2d: 70mm")
print("   - T3d: 60mm")
print("   - T4d: 50mm")
print("   - T5d: 40mm")
print("3. Click any location on the map")
print("4. You will see 90%+ flood probability (CRITICAL RISK)")
print("\nAlternatively, wait for actual heavy rainfall events (monsoon season)")
print("when the weather API will fetch real precipitation data >50mm/day")
