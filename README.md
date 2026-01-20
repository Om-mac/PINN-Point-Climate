# PINN-Point Climate - Stage 1: Binary Flood Classification

🌍 **Interactive Web Application** for real-time flood risk prediction across India using Machine Learning and Real-Time Weather Data.

**Quick Links**: [Quick Start](QUICKSTART.md) | [Technical Docs](TECHNICAL_DOCS.md) | [Project Summary](PROJECT_SUMMARY.md) | [Map Integration Guide](MAP_INTEGRATION_GUIDE.md) | [Weather API Setup](WEATHER_API_SETUP.md)

---

## 🎯 Project Overview

This system predicts flood probability for any location in India using:
- ✅ **Random Forest ML Model** (100% accuracy, 1.0 ROC-AUC)
- 🗺️ **Interactive Google Maps Interface** for coordinate selection
- 📡 **Real-Time Weather API Integration** (Visual Crossing)
- 🇮🇳 **India-Specific** (trained on INDOFLOODS dataset)
- ⚠️ **PINN Stage 2 Trigger** when probability > 70%

### 🆕 Latest Updates (January 2026)

#### ✨ New Features
- **Interactive Map Interface**: Click anywhere on the map to get flood predictions
- **Real-Time Weather Data**: Automatic precipitation fetching via Visual Crossing API
- **Geographic Validation**: Restricts predictions to India only (6°N to 37°N, 68°E to 97.5°E)
- **Boundary Value Analysis**: Properly handles zero precipitation (dry season)
- **Distance Warnings**: Alerts when selected location is >300km from nearest station

#### 🔧 Model Improvements
- **Enhanced Training Data**: 30% explicit zero-precipitation samples for dry season accuracy
- **Removed Location Bias**: Excluded Lat/Lon features to prevent location overfitting
- **Precipitation-Focused**: Model now prioritizes rainfall patterns over geographic location
- **Realistic Predictions**: 0mm rainfall → 0.5% risk, 30mm+ → 90%+ risk

#### 🌐 Web Application
- **Flask Backend**: REST API for predictions at `/api/predict`
- **Google Maps Integration**: Visual station markers and location selection
- **Manual & Automatic Modes**: Choose between manual precipitation entry or API fetch
- **Responsive UI**: Real-time predictions with color-coded risk levels

---

## 📊 Dataset

The INDOFLOODS dataset contains:
- **Catchment Characteristics** (155 stations): Static geomorphological features
- **Flood Events** (4,548 events): Historical flood records with dates and severity
- **Metadata** (219 stations): Station information, coordinates, operational periods
- **Precipitation Variables** (4,548 records): Daily precipitation data (T1d to T10d)

## 🔧 Features

### 1. Antecedent Precipitation Index (API)
- **API_3day**: Sum of rainfall from 3 days before target date
- **API_7day**: Sum of rainfall from 7 days before target date
- **API_10day**: Sum of rainfall from 10 days before target date

### 2. Geomorphological Features
- Upstream Catchment Area
- Elevation (Catchment Relief)
- Slope (Relief Ratio)
- Drainage Density
- Stream Order
- Catchment Length

### 3. Categorical Features
- Land cover type
- Soil type
- Lithology type
- Köppen-Geiger Climate Type

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd PINN-Point-Climate

# Install dependencies
pip install -r requirements.txt
```

### 2. Set Up API Keys

```bash
# Set Visual Crossing API key (for real-time weather)
export VISUALCROSSING_API_KEY='your_api_key_here'

# Google Maps API key is already configured in templates/map_interface.html
```

### 3. Run the Application

**Option A: Web Application (Recommended)**
```bash
# Start Flask server
VISUALCROSSING_API_KEY='your_key' python app.py

# Open browser to http://localhost:5000
```

**Option B: Train Model Only**
```bash
# Random Forest (default)
python flood_prediction_model.py

# XGBoost (alternative)
python flood_prediction_xgboost.py

# Compare both models
python compare_models.py
```

---

## 🗺️ Web Application Features

### Interactive Map Interface

1. **Click on Map**: Select any location within India
2. **View Station Info**: See nearest flood monitoring station
3. **Choose Data Source**:
   - ✅ **Fetch Real Weather Data**: Automatic API call for last 10 days
   - ✏️ **Manual Entry**: Enter custom precipitation values
4. **Get Prediction**: Instant flood probability with risk level

### API Endpoints

#### `GET /`
- Main web interface with Google Maps

#### `GET /api/stations`
- Returns all INDOFLOODS station locations

#### `POST /api/predict`
```json
{
  "lat": 28.5,
  "lon": 77.8,
  "precipitation": {
    "T1d": 85.5,
    "T2d": 70.3
  },
  "use_real_weather": true
}
```

**Response:**
```json
{
  "success": true,
  "prediction": {
    "probability": 0.923,
    "probability_percent": "92.3%",
    "prediction": "FLOOD",
    "risk_level": "CRITICAL",
    "trigger_pinn": true
  },
  "location": {
    "station_name": "Hoshangabad",
    "distance_km": "105.44",
    "river": "Narmada",
    "state": "Madhya Pradesh",
    "warning": null
  }
}
```

---

## 🧪 Model Performance

### Current Results (After Optimization)

| Metric | Value |
|--------|-------|
| **Accuracy** | 100% |
| **ROC-AUC Score** | 1.0000 |
| **Precision (Flood)** | 1.00 |
| **Recall (Flood)** | 1.00 |
| **F1-Score** | 1.00 |

### Feature Importance (Top 10)

| Feature | Importance |
|---------|-----------|
| API_10day | 18.02% |
| T10d | 15.29% |
| T7d | 14.39% |
| T9d | 12.66% |
| T8d | 11.27% |
| API_7day | 7.12% |
| T6d | 7.11% |
| T5d | 5.30% |
| T4d | 3.75% |
| T3d | 3.70% |

**Key Insight**: Model now correctly prioritizes **precipitation features** (86%+ total importance) over geographic features (<1%).

### Training Data Composition

- **Total Samples**: 6,098
  - ✅ **Flood Events**: 4,548 (74.6%)
  - ❌ **Non-Flood**: 1,550 (25.4%)
    - 30% with 0mm precipitation (dry season)
    - 30% with 0-5mm (light rain)
    - 40% with 5-20mm (moderate rain, no flood)

### Boundary Value Analysis

| Precipitation | Expected Risk | Actual Result |
|--------------|---------------|---------------|
| 0mm (all days) | <5% | ✅ 0.5% (LOW) |
| 5mm/day (50mm total) | <5% | ✅ 0.2% (LOW) |
| 15mm/day (150mm total) | <5% | ✅ 0.0% (LOW) |
| 30mm/day (300mm total) | >90% | ✅ 97.5% (CRITICAL) |
| 60mm/day (600mm total) | >95% | ✅ 100% (CRITICAL) |

---

## 🌍 Geographic Restrictions

The model is trained exclusively on **Indian river flood data** and only works within India:

**Valid Region:**
- Latitude: 6.0°N to 37.0°N
- Longitude: 68.0°E to 97.5°E

**What Happens Outside India:**
- API returns HTTP 400 error
- Alert message: "Location outside India. This model is trained on INDOFLOODS dataset and only works for locations within India."

**Distance Warnings:**
- If selected location is >300km from nearest station, a warning is displayed
- Prediction accuracy may be lower for remote areas

---

## 📁 Project Structure

```
PINN-Point-Climate/
├── app.py                          # Flask web application
├── flood_prediction_model.py       # Main ML model (Random Forest)
├── flood_prediction_xgboost.py     # Alternative XGBoost model
├── compare_models.py               # Model comparison script
├── weather_api.py                  # Weather API integration
├── test_boundary_values.py         # Boundary analysis testing
├── templates/
│   └── map_interface.html          # Google Maps web interface
├── models/
│   └── flood_prediction_rf.pkl     # Trained Random Forest model
├── DATA/
│   ├── catchment_characteristics_indofloods.csv
│   ├── floodevents_indofloods.csv
│   ├── metadata_indofloods.csv
│   └── precipitation_variables_indofloods.csv
├── requirements.txt                # Python dependencies
└── *.md                           # Documentation files
```

---

## 🔑 API Keys Required

### 1. Visual Crossing Weather API
- **Purpose**: Fetch real-time precipitation data
- **Get Free Key**: https://www.visualcrossing.com/
- **Free Tier**: 1000 requests/day
- **Setup**: `export VISUALCROSSING_API_KEY='your_key'`

### 2. Google Maps JavaScript API
- **Purpose**: Interactive map interface
- **Get Key**: https://console.cloud.google.com/
- **Already Configured**: Key in `templates/map_interface.html` line 556
- **APIs Needed**: Maps JavaScript API

---

## 🛠️ Technical Stack

- **Backend**: Flask 2.3.0
- **ML**: scikit-learn 1.2.0, XGBoost 1.7.0
- **Data**: pandas, numpy, scipy
- **Visualization**: matplotlib, seaborn
- **APIs**: Google Maps JavaScript API, Visual Crossing Weather API

---

## 📝 How It Works

1. **User clicks location** on Google Maps
2. **Coordinates sent** to Flask backend (`/api/predict`)
3. **Geographic validation**: Check if within India
4. **Find nearest station** from 155 INDOFLOODS stations
5. **Fetch weather data**: 
   - If checkbox enabled → Visual Crossing API (real precipitation)
   - If disabled → Use manual input or defaults
6. **Calculate features**:
   - API_3day, API_7day, API_10day
   - 10-day precipitation (T1d to T10d)
   - Station characteristics (drainage, relief, etc.)
7. **Model prediction**: Random Forest outputs probability
8. **Risk classification**:
   - CRITICAL (≥70%) - Red - Triggers PINN Stage 2
   - HIGH (50-70%) - Orange
   - MODERATE (30-50%) - Yellow
   - LOW (<30%) - Green
9. **Display results** with station info, river name, distance

---

## 🚨 PINN Stage 2 Trigger

When flood probability **exceeds 70%**, the system recommends activating **Physics-Informed Neural Network (PINN) Stage 2** for:
- Detailed hydraulic simulation
- Flood extent mapping
- Water depth prediction
- Inundation timing analysis

This two-stage approach optimizes computational resources by running complex physics simulations only for high-risk scenarios.

---

## 🧪 Testing & Validation

### Run Tests

```bash
# Boundary value analysis
python test_boundary_values.py

# Test with actual API values
python test_actual_api_values.py

# Compare model performance
python compare_models.py
```

### Example Predictions

```bash
# Make predictions with example data
python example_prediction.py
```

---

## 📚 Documentation

- **[QUICKSTART.md](QUICKSTART.md)**: Get started in 5 minutes
- **[TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)**: Detailed technical specifications
- **[MAP_INTEGRATION_GUIDE.md](MAP_INTEGRATION_GUIDE.md)**: Google Maps setup
- **[WEATHER_API_SETUP.md](WEATHER_API_SETUP.md)**: Weather API configuration
- **[PROJECT_SUMMARY.md](PROJECT_SUMMARY.md)**: Complete project overview

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 📄 License

This project is part of the INDOFLOODS dataset research initiative for flood prediction in India.

---

## 🙏 Acknowledgments

- **INDOFLOODS Dataset**: Indian flood monitoring stations data
- **Visual Crossing**: Weather API for real-time precipitation
- **Google Maps**: Interactive mapping interface
- **scikit-learn**: Machine learning framework

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

## 🔮 Future Enhancements

- [ ] Stage 2 PINN physics-based simulation
- [ ] Mobile application (iOS/Android)
- [ ] SMS/Email flood alerts
- [ ] Historical flood event visualization
- [ ] Multi-day forecast predictions (7-14 days)
- [ ] Integration with more weather APIs
- [ ] Satellite imagery analysis
- [ ] Real-time sensor data integration
1. Load all CSV files from `DATA/` directory
2. Merge datasets using station IDs (GaugeID)
3. Generate negative samples (non-flood dates)
4. Create API and geomorphological features
5. Train Random Forest classifier
6. Evaluate model performance
7. Save model to `models/flood_prediction_rf.pkl`
8. Generate visualization (`flood_model_evaluation.png`)

### Use the Trained Model

```bash
python example_prediction.py
```

This demonstrates:
- Loading the trained model
- Making predictions for new station data
- PINN Stage 2 trigger logic
- Risk level assessment

### Programmatic Usage

```python
import pickle
import pandas as pd

# Load model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

# Prepare features
station_features = {
    'T1d': 85.5,
    'T2d': 120.3,
    'API_3day': 301.6,
    'Drainage Area': 3542.1,
    # ... other features
}

feature_df = pd.DataFrame([station_features])[feature_columns]

# Predict
probability = model.predict_proba(feature_df)[0, 1]
trigger_pinn = probability > 0.7

print(f"Flood Probability: {probability:.3f}")
print(f"Trigger PINN Stage 2: {trigger_pinn}")
```

## 📈 Model Architecture

**Algorithm**: Random Forest Classifier
- **n_estimators**: 200 trees
- **max_depth**: 15
- **class_weight**: balanced (handles imbalanced dataset)
- **Output**: Probability score (0.0 to 1.0)

## 🔄 PINN Integration Workflow

```
┌─────────────────────────────────────┐
│  Stage 1: Random Forest Classifier  │
│  • Fast binary classification       │
│  • Output: Flood probability        │
└──────────────┬──────────────────────┘
               │
               ▼
        Probability > 0.7?
               │
         ┌─────┴─────┐
         │           │
        YES          NO
         │           │
         ▼           ▼
    ┌────────┐   Continue
    │ Stage 2│   Monitoring
    │  PINN  │
    │Simulate│
    └────────┘
```

## 📊 Model Performance

After training, the model provides:
- **Classification Report**: Precision, Recall, F1-Score
- **Confusion Matrix**: True/False Positives/Negatives
- **ROC-AUC Score**: Model discrimination ability
- **Feature Importance**: Top contributing features
- **Probability Distribution**: Risk stratification

## 🎯 Key Features

1. **Negative Sample Generation**: Automatically creates non-flood samples for balanced training
2. **Feature Engineering**: API features capture temporal rainfall patterns
3. **Multi-source Integration**: Combines static (catchment) and dynamic (precipitation) features
4. **PINN Trigger**: Threshold-based activation (>0.7) for detailed simulation
5. **Categorical Encoding**: Handles soil, land cover, and climate types

## 📁 Project Structure

```
PINN-Point-Climate/
├── DATA/                                   # Input datasets
│   ├── catchment_characteristics_indofloods.csv
│   ├── floodevents_indofloods.csv
│   ├── metadata_indofloods.csv
│   └── precipitation_variables_indofloods.csv
│
├── models/                                 # Trained models
│   ├── flood_prediction_rf.pkl            # Random Forest model
│   └── flood_prediction_xgb.pkl           # XGBoost model
│
├── Documentation/
│   ├── README.md                          # This file (main docs)
│   ├── QUICKSTART.md                      # 5-minute tutorial
│   ├── TECHNICAL_DOCS.md                  # Technical details
│   ├── PROJECT_SUMMARY.md                 # Executive summary
│   └── INDEX.md                           # Documentation index
│
├── Scripts/
│   ├── flood_prediction_model.py          # Train Random Forest
│   ├── flood_prediction_xgboost.py        # Train XGBoost
│   ├── compare_models.py                  # Compare RF vs XGBoost
│   └── example_prediction.py              # Demo predictions
│
├── requirements.txt                        # Dependencies
└── flood_model_evaluation.png             # Visualizations
```

## 🔬 Next Steps (Stage 2)

When flood probability > 0.7:
1. **Trigger PINN Simulation**
2. **Physics-Based Modeling**: Solve shallow water equations
3. **High-Resolution Output**: Detailed flood extent, depth, velocity
4. **Parameter Estimation**: Optimal values for flood management
5. **Real-time Forecasting**: Integrate with weather forecasts

## 📝 Notes

- The model uses **balanced class weights** to handle the imbalanced dataset (more flood events than non-flood events)
- **Negative samples** are generated by randomly selecting dates where no flood occurred within a 7-day window
- **Missing precipitation data** for negative samples is filled using station-specific averages
- The **0.7 threshold** for PINN trigger can be adjusted based on operational requirements

## 🤝 Contributing

This is Stage 1 of the PINN-Point Climate project. Future stages will integrate:
- Physics-Informed Neural Networks (PINN)
- Real-time weather data ingestion
- Spatial flood extent modeling
- Decision support system for flood management

## 📄 License

[Add license information]

## 📧 Contact

[Add contact information]
