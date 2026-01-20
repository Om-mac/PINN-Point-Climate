"""
Test with ACTUAL values being sent to the model from the API
"""
import pickle
import pandas as pd

# Load model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

# Load sample station
stations_df = pd.read_csv('DATA/catchment_characteristics_indofloods.csv')
metadata_df = pd.read_csv('DATA/metadata_indofloods.csv')
sample = stations_df.merge(metadata_df, on='GaugeID').iloc[0]

print("Testing with ACTUAL API conversion values:")
print("=" * 60)

# What the API is actually sending after safe_float conversion
# 0mm → 0.01mm (per safe_float logic: max(0.01, 0))
precip = [0.01] * 10

features = {}
for i, val in enumerate(precip, 1):
    features[f'T{i}d'] = val

features['API_3day'] = sum(precip[:3])
features['API_7day'] = sum(precip[:7])
features['API_10day'] = sum(precip[:10])

# Add station features
features['Drainage Area'] = sample.get('Drainage Area', 0)
features['Catchment Relief'] = sample.get('Catchment Relief', 0)
features['Catchment Length'] = sample.get('Catchment Length', 0)
features['Relief Ratio'] = sample.get('Relief Ratio', 0)
features['Drainage Density'] = sample.get('Drainage Density', 0)
features['Stream Order'] = sample.get('Stream Order', 0)
features['Catchment Area'] = sample.get('Catchment Area', 0)

# Encoded features
for col in feature_columns:
    if col.endswith('_encoded') and col not in features:
        features[col] = 0

# Predict
feature_vector = [features.get(col, 0) for col in feature_columns]
X = pd.DataFrame([feature_vector], columns=feature_columns)
probability = model.predict_proba(X)[0][1]

print(f"Precipitation: {precip[0]:.2f}mm for all 10 days")
print(f"API_3day: {features['API_3day']:.2f}mm")
print(f"API_7day: {features['API_7day']:.2f}mm")
print(f"API_10day: {features['API_10day']:.2f}mm")
print(f"\n🎯 Flood Probability: {probability*100:.1f}%")

if probability > 0.3:
    print("❌ PROBLEM: Still showing 30%+ risk with near-zero precipitation!")
else:
    print("✅ FIXED: Low precipitation = Low risk")
