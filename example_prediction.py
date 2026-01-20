"""
Example script demonstrating how to use the trained flood prediction model
for real-time predictions and PINN Stage 2 integration.
"""

import pickle
import pandas as pd
import numpy as np


def load_model(filepath='models/flood_prediction_rf.pkl'):
    """Load the trained model."""
    with open(filepath, 'rb') as f:
        model_data = pickle.load(f)
    return model_data


def predict_flood_risk(model_data, station_features):
    """
    Predict flood risk for a station.
    
    Args:
        model_data: Loaded model dictionary
        station_features: Dictionary with feature values
        
    Returns:
        Prediction results with PINN trigger status
    """
    model = model_data['model']
    feature_columns = model_data['feature_columns']
    probability_threshold = model_data['probability_threshold']
    
    # Create feature vector - ensure all required features are present
    feature_dict = {}
    for col in feature_columns:
        if col in station_features:
            feature_dict[col] = station_features[col]
        else:
            # Use default value for missing features
            feature_dict[col] = 0.0
    
    feature_df = pd.DataFrame([feature_dict])
    
    # Predict
    probability = model.predict_proba(feature_df)[0, 1]
    prediction = int(probability > 0.5)
    trigger_pinn = probability > probability_threshold
    
    return {
        'flood_probability': float(probability),
        'prediction': 'FLOOD' if prediction == 1 else 'NO FLOOD',
        'trigger_pinn_stage2': trigger_pinn,
        'confidence': 'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM',
        'risk_level': get_risk_level(probability)
    }


def get_risk_level(probability):
    """Determine risk level based on probability."""
    if probability >= 0.7:
        return "CRITICAL"
    elif probability >= 0.5:
        return "HIGH"
    elif probability >= 0.3:
        return "MODERATE"
    else:
        return "LOW"


def main():
    """Demonstration of model usage."""
    print("="*60)
    print("FLOOD PREDICTION - EXAMPLE USAGE")
    print("="*60)
    
    # Load trained model
    print("\nLoading trained model...")
    model_data = load_model('models/flood_prediction_rf.pkl')
    print("✓ Model loaded successfully")
    
    # Example station data (replace with actual real-time data)
    print("\n" + "-"*60)
    print("EXAMPLE 1: High-risk scenario (heavy recent rainfall)")
    print("-"*60)
    
    station_data_high_risk = {
        # Recent precipitation (T1d = yesterday, T2d = 2 days ago, etc.)
        'T1d': 85.5,
        'T2d': 120.3,
        'T3d': 95.8,
        'T4d': 110.2,
        'T5d': 98.4,
        'T6d': 105.6,
        'T7d': 88.9,
        'T8d': 92.1,
        'T9d': 78.3,
        'T10d': 82.7,
        
        # API features (automatically calculated if not provided)
        'API_3day': 301.6,  # T1d + T2d + T3d
        'API_7day': 704.7,  # Sum of T1d to T7d
        'API_10day': 957.8, # Sum of T1d to T10d
        
        # Geomorphological features
        'Drainage Area': 3542.1,
        'Catchment Relief': 4416.0,
        'Catchment Length': 116870.2,
        'Relief Ratio': 0.0378,
        'Drainage Density': 0.000192,
        'Stream Order': 4.0,
        'Catchment Area': 3542.1,
        'Latitude': 28.5,
        'Longitude': 77.8,
        
        # Encoded categorical features (use label encoders from model)
        'Land cover_encoded': 1,
        'Soil type_encoded': 0,
        'lithology type_encoded': 2,
        'KoppenGeiger Climate Type_encoded': 1
    }
    
    result = predict_flood_risk(model_data, station_data_high_risk)
    
    print(f"\nPrediction Results:")
    print(f"  Flood Probability: {result['flood_probability']:.3f} ({result['flood_probability']*100:.1f}%)")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Trigger PINN Stage 2: {'YES ⚠️' if result['trigger_pinn_stage2'] else 'NO'}")
    
    if result['trigger_pinn_stage2']:
        print(f"\n  ⚠️  HIGH RISK DETECTED!")
        print(f"  → Initiating Stage 2 PINN Simulation for detailed flood modeling")
        print(f"  → PINN will simulate detailed hydraulic parameters")
    
    # Example 2: Low-risk scenario
    print("\n" + "-"*60)
    print("EXAMPLE 2: Low-risk scenario (minimal rainfall)")
    print("-"*60)
    
    station_data_low_risk = {
        'T1d': 5.2,
        'T2d': 8.3,
        'T3d': 3.1,
        'T4d': 6.5,
        'T5d': 4.8,
        'T6d': 7.2,
        'T7d': 5.9,
        'T8d': 6.1,
        'T9d': 4.3,
        'T10d': 5.7,
        'API_3day': 16.6,
        'API_7day': 41.0,
        'API_10day': 57.1,
        'Drainage Area': 3542.1,
        'Catchment Relief': 4416.0,
        'Catchment Length': 116870.2,
        'Relief Ratio': 0.0378,
        'Drainage Density': 0.000192,
        'Stream Order': 4.0,
        'Catchment Area': 3542.1,
        'Latitude': 28.5,
        'Longitude': 77.8,
        'Land cover_encoded': 1,
        'Soil type_encoded': 0,
        'lithology type_encoded': 2,
        'KoppenGeiger Climate Type_encoded': 1
    }
    
    result = predict_flood_risk(model_data, station_data_low_risk)
    
    print(f"\nPrediction Results:")
    print(f"  Flood Probability: {result['flood_probability']:.3f} ({result['flood_probability']*100:.1f}%)")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Confidence: {result['confidence']}")
    print(f"  Trigger PINN Stage 2: {'YES ⚠️' if result['trigger_pinn_stage2'] else 'NO'}")
    
    print("\n" + "="*60)
    print("PINN INTEGRATION WORKFLOW")
    print("="*60)
    print("\nStage 1 (This Model):")
    print("  • Binary classification using Random Forest")
    print("  • Output: Flood probability (0.0 to 1.0)")
    print("  • Fast screening of flood risk")
    print("\nStage 2 (PINN - If probability > 0.7):")
    print("  • Physics-Informed Neural Network simulation")
    print("  • Detailed hydraulic modeling")
    print("  • High-resolution flood extent prediction")
    print("  • Parameter estimation for flood management")
    print("\n" + "="*60)


if __name__ == "__main__":
    main()
