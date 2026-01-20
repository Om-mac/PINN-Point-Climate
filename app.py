"""
Flask Web Application with Google Maps Integration
Click on any location to predict flood probability for that area.
"""

from flask import Flask, render_template, request, jsonify
import pickle
import pandas as pd
import numpy as np
from scipy.spatial import distance
from weather_api import fetch_real_precipitation
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Get API keys from environment variables
GOOGLE_MAPS_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY')
VISUALCROSSING_API_KEY = os.getenv('VISUALCROSSING_API_KEY')

if not GOOGLE_MAPS_API_KEY:
    raise ValueError("GOOGLE_MAPS_API_KEY not found in environment variables. Please set it in .env file.")
if not VISUALCROSSING_API_KEY:
    raise ValueError("VISUALCROSSING_API_KEY not found in environment variables. Please set it in .env file.")

# Load the trained model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']
probability_threshold = model_data['probability_threshold']

# Load station data for nearest neighbor matching
metadata_df = pd.read_csv('DATA/metadata_indofloods.csv')
catchment_df = pd.read_csv('DATA/catchment_characteristics_indofloods.csv')

# Merge for complete station information
stations_df = metadata_df.merge(catchment_df, on='GaugeID', how='inner')

# India geographic boundaries (approximate)
INDIA_BOUNDS = {
    'lat_min': 6.0,   # Southernmost point (Indira Point, Nicobar)
    'lat_max': 37.0,  # Northernmost point (Siachen Glacier)
    'lon_min': 68.0,  # Westernmost point (Gujarat border)
    'lon_max': 97.5   # Easternmost point (Arunachal Pradesh)
}

MAX_DISTANCE_KM = 300  # Maximum distance from nearest station (km)


def is_within_india(lat, lon):
    """Check if coordinates are within India's geographic boundaries."""
    return (INDIA_BOUNDS['lat_min'] <= lat <= INDIA_BOUNDS['lat_max'] and
            INDIA_BOUNDS['lon_min'] <= lon <= INDIA_BOUNDS['lon_max'])


def find_nearest_station(lat, lon):
    """Find the nearest station to given coordinates."""
    coords = np.array([[lat, lon]])
    station_coords = stations_df[['Latitude', 'Longitude']].values
    
    # Calculate distances
    distances = distance.cdist(coords, station_coords, 'euclidean')[0]
    nearest_idx = np.argmin(distances)
    
    nearest_station = stations_df.iloc[nearest_idx]
    nearest_distance = distances[nearest_idx] * 111  # Convert to km (approximate)
    
    return nearest_station, nearest_distance


def prepare_features_for_prediction(station_data, recent_precipitation):
    """
    Prepare features for prediction.
    
    Args:
        station_data: Station information from nearest match
        recent_precipitation: Dictionary with T1d to T10d values
    """
    # Helper function to safely convert to float, handling None values
    def safe_float(value, default=0.0):
        """Convert to float, default to 0.0 for missing values."""
        if value is None or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Get precipitation values with None handling
    precip_values = {}
    for i in range(1, 11):
        key = f'T{i}d'
        precip_values[key] = safe_float(recent_precipitation.get(key))
    
    # Calculate API features
    api_3day = sum([precip_values[f'T{i}d'] for i in range(1, 4)])
    api_7day = sum([precip_values[f'T{i}d'] for i in range(1, 8)])
    api_10day = sum([precip_values[f'T{i}d'] for i in range(1, 11)])
    
    # Build feature dictionary
    features = {
        'API_3day': api_3day,
        'API_7day': api_7day,
        'API_10day': api_10day,
        'T1d': precip_values['T1d'],
        'T2d': precip_values['T2d'],
        'T3d': precip_values['T3d'],
        'T4d': precip_values['T4d'],
        'T5d': precip_values['T5d'],
        'T6d': precip_values['T6d'],
        'T7d': precip_values['T7d'],
        'T8d': precip_values['T8d'],
        'T9d': precip_values['T9d'],
        'T10d': precip_values['T10d'],
        'Drainage Area': station_data.get('Drainage Area', 0),
        'Catchment Relief': station_data.get('Catchment Relief', 0),
        'Catchment Length': station_data.get('Catchment Length', 0),
        'Relief Ratio': station_data.get('Relief Ratio', 0),
        'Drainage Density': station_data.get('Drainage Density', 0),
        'Stream Order': station_data.get('Stream Order', 0),
        'Catchment Area': station_data.get('Catchment Area', 0),
    }
    
    # Add encoded categorical features (use defaults if not available)
    for col in feature_columns:
        if col.endswith('_encoded') and col not in features:
            features[col] = 0  # Default encoding
    
    return features


@app.route('/')
def index():
    """Render the main map interface with API key."""
    return render_template('map_interface.html', google_maps_api_key=GOOGLE_MAPS_API_KEY)


@app.route('/api/stations')
def get_stations():
    """Get all station locations for map markers."""
    stations = []
    for _, row in stations_df.iterrows():
        stations.append({
            'id': row['GaugeID'],
            'name': row.get('Station', 'Unknown'),
            'lat': float(row['Latitude']),
            'lon': float(row['Longitude']),
            'river': row.get('River Name/ Tributory/ SubTributory', 'Unknown'),
            'state': row.get('State', 'Unknown')
        })
    return jsonify(stations)


@app.route('/api/predict', methods=['POST'])
def predict():
    """
    Predict flood probability for given coordinates.
    
    Expected JSON:
    {
        "lat": 28.5,
        "lon": 77.8,
        "precipitation": {
            "T1d": 85.5,
            "T2d": 120.3,
            ...
        },
        "use_real_weather": true/false  (optional, default: false)
    }
    """
    try:
        data = request.json
        lat = float(data['lat'])
        lon = float(data['lon'])
        precipitation = data.get('precipitation', {})
        use_real_weather = data.get('use_real_weather', False)
        
        # Validate location is within India
        if not is_within_india(lat, lon):
            return jsonify({
                'error': 'Location outside India',
                'message': f'Coordinates ({lat:.4f}, {lon:.4f}) are outside India. This model is trained on INDOFLOODS dataset and only works for locations within India.',
                'india_bounds': INDIA_BOUNDS
            }), 400
        
        # If use_real_weather is True, fetch from API
        if use_real_weather:
            print(f"📡 Fetching real weather data for {lat}, {lon}...")
            precipitation = fetch_real_precipitation(lat, lon, api_provider='visualcrossing')
            print(f"   Retrieved precipitation (mm): {precipitation}")
            data_source = "Real Weather API"
        # If no precipitation data provided, use moderate default values
        elif not precipitation or all(v is None for v in precipitation.values()):
            precipitation = {
                'T1d': 50.0, 'T2d': 45.0, 'T3d': 40.0, 'T4d': 38.0, 'T5d': 42.0,
                'T6d': 48.0, 'T7d': 35.0, 'T8d': 32.0, 'T9d': 30.0, 'T10d': 28.0
            }
            data_source = "Default Values"
        else:
            data_source = "Manual Entry"
        
        # Find nearest station
        nearest_station, distance_km = find_nearest_station(lat, lon)
        
        # Warn if nearest station is too far
        if distance_km > MAX_DISTANCE_KM:
            warning = f"⚠️ Nearest station is {distance_km:.1f} km away. Prediction may be less accurate for remote areas."
        else:
            warning = None
        
        # Prepare features
        station_dict = nearest_station.to_dict()
        features = prepare_features_for_prediction(station_dict, precipitation)
        
        # Debug: Print calculated API indices
        api_3 = features.get('API_3day', 0)
        api_7 = features.get('API_7day', 0)
        api_10 = features.get('API_10day', 0)
        print(f"   Calculated API indices: 3day={api_3:.2f}, 7day={api_7:.2f}, 10day={api_10:.2f}")
        
        # Ensure all required features are present
        feature_dict = {}
        for col in feature_columns:
            feature_dict[col] = features.get(col, 0.0)
        
        # Make prediction
        feature_df = pd.DataFrame([feature_dict])
        probability = float(model.predict_proba(feature_df)[0, 1])
        prediction = int(probability > 0.5)
        trigger_pinn = probability > probability_threshold
        
        # Determine risk level
        if probability >= 0.7:
            risk_level = "CRITICAL"
            risk_color = "#dc3545"
        elif probability >= 0.5:
            risk_level = "HIGH"
            risk_color = "#ff9800"
        elif probability >= 0.3:
            risk_level = "MODERATE"
            risk_color = "#ffc107"
        else:
            risk_level = "LOW"
            risk_color = "#28a745"
        
        response = {
            'success': True,
            'prediction': {
                'probability': probability,
                'probability_percent': f"{probability * 100:.1f}%",
                'prediction': 'FLOOD' if prediction == 1 else 'NO FLOOD',
                'risk_level': risk_level,
                'risk_color': risk_color,
                'trigger_pinn': trigger_pinn,
                'confidence': 'HIGH' if probability > 0.7 or probability < 0.3 else 'MEDIUM'
            },
            'location': {
                'clicked_lat': lat,
                'clicked_lon': lon,
                'nearest_station': nearest_station['GaugeID'],
                'station_name': nearest_station.get('Station', 'Unknown'),
                'station_id': nearest_station.get('GaugeID', 'Unknown'),
                'station_lat': float(nearest_station['Latitude']),
                'station_lon': float(nearest_station['Longitude']),
                'distance_km': f"{distance_km:.2f}",
                'river': nearest_station.get('River Name/ Tributory/ SubTributory', 'Unknown'),
                'state': nearest_station.get('State', 'Unknown'),
                'warning': warning
            },
            'precipitation': precipitation,
            'data_source': data_source
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/predict-with-weather', methods=['POST'])
def predict_with_weather():
    """
    Predict flood probability using real weather data.
    This endpoint can be extended to fetch real-time weather data from APIs.
    """
    data = request.json
    lat = float(data['lat'])
    lon = float(data['lon'])
    
    # TODO: Integrate with weather API (e.g., OpenWeatherMap) to get real precipitation
    # For now, use sample data
    
    return predict()


if __name__ == '__main__':
    print("\n" + "="*60)
    print("🗺️  PINN-Point Climate - Interactive Map Interface")
    print("="*60)
    print("\n📍 Starting Flask server...")
    print("🌐 Open your browser and go to: http://localhost:5000")
    print("\n📝 How to use:")
    print("   1. Click anywhere on the map")
    print("   2. Enter recent precipitation data (optional)")
    print("   3. Click 'Predict Flood Risk'")
    print("   4. View probability and risk level")
    print("\n⚠️  Press Ctrl+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
