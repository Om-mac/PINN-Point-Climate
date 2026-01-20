# PINN-Point Climate Stage 1 - Technical Documentation

## Overview

This document provides detailed technical information about the Stage 1 flood prediction model.

## Data Schema

### 1. Catchment Characteristics (155 stations)
- **Key Column**: `GaugeID`
- **Type**: Static features (time-invariant)
- **Features**: 
  - Geomorphological: Stream Order, Drainage Area, Catchment Relief, etc.
  - Climate: Annual Mean Temperature, Precipitation metrics
  - Socioeconomic: GDP, HDI, Population Density
  - Physical: Land cover, Soil type, Lithology type

### 2. Flood Events (4,548 events)
- **Key Column**: `EventID` (format: INDOFLOODS-gauge-XXXX-N)
- **Derived Column**: `GaugeID` (extracted from EventID)
- **Type**: Historical flood records
- **Key Fields**:
  - Start Date, End Date
  - Peak Flood Level (m)
  - Peak FL Date (used as target date)
  - Flood Type (Flood, Severe Flood, etc.)

### 3. Metadata (214 stations)
- **Key Column**: `GaugeID`
- **Type**: Station information
- **Key Fields**:
  - Latitude, Longitude
  - River Name, Basin, State
  - Warning Level, Danger Level
  - Start_date, End_date (operational period)
  - Catchment Area

### 4. Precipitation Variables (4,548 records)
- **Key Column**: `EventID`
- **Type**: Dynamic features (time-varying)
- **Features**: T1d to T10d (daily precipitation)
  - T1d: Precipitation 1 day before event
  - T2d: Precipitation 2 days before event
  - ... T10d: Precipitation 10 days before event

## Data Processing Pipeline

### Step 1: Load Data
```python
model.load_data()
```
- Loads all 4 CSV files
- Initial validation of data integrity

### Step 2: Prepare Flood Events (Positive Samples)
```python
model.prepare_flood_events()
```
- Extracts GaugeID from EventID
- Converts dates to datetime format
- Creates Flood_Label = 1 for all flood events
- Uses Peak FL Date as target date

### Step 3: Generate Negative Samples
```python
model.generate_negative_samples(samples_per_station=10)
```
- For each station with flood records:
  - Randomly selects dates within operational period
  - Ensures dates are NOT within 7 days of any flood event
  - Creates Flood_Label = 0
  - Generates EventID: INDOFLOODS-gauge-XXXX-NEG-N

**Rationale**: Dataset is imbalanced (only flood events), need non-flood samples for binary classification

### Step 4: Feature Engineering
```python
model.create_features()
```

#### Antecedent Precipitation Index (API)
- **API_3day**: Sum of T1d + T2d + T3d
- **API_7day**: Sum of T1d to T7d
- **API_10day**: Sum of T1d to T10d

**Rationale**: Captures cumulative rainfall effect on flood risk

#### Feature Merging
1. Merge precipitation data (EventID)
2. For negative samples without precipitation:
   - Use station-specific average precipitation
   - Fallback to overall mean if station average unavailable
3. Merge catchment characteristics (GaugeID)
4. Merge metadata (GaugeID)

#### Categorical Encoding
- Land cover → Label Encoding
- Soil type → Label Encoding
- Lithology type → Label Encoding
- Köppen-Geiger Climate Type → Label Encoding

#### Final Feature Set (26 features)
1. **API Features** (3): API_3day, API_7day, API_10day
2. **Temporal Precipitation** (10): T1d to T10d
3. **Geomorphological** (9): Drainage Area, Catchment Relief, Catchment Length, Relief Ratio, Drainage Density, Stream Order, Catchment Area, Latitude, Longitude
4. **Categorical Encoded** (4): Land cover, Soil type, Lithology, Climate Type

### Step 5: Train Model
```python
model.train_model(test_size=0.2, random_state=42)
```

#### Model Configuration
- **Algorithm**: Random Forest Classifier
- **Hyperparameters**:
  - n_estimators: 200 (number of decision trees)
  - max_depth: 15 (maximum tree depth)
  - min_samples_split: 10 (minimum samples to split node)
  - min_samples_leaf: 5 (minimum samples per leaf)
  - class_weight: 'balanced' (handles class imbalance)
  - n_jobs: -1 (use all CPU cores)
  - random_state: 42 (reproducibility)

#### Data Split
- **Training**: 80% (stratified by flood label)
- **Test**: 20% (stratified by flood label)

### Step 6: Model Evaluation
```python
model.evaluate_model()
```

#### Metrics
1. **Classification Report**: Precision, Recall, F1-Score
2. **Confusion Matrix**: True/False Positives/Negatives
3. **ROC-AUC Score**: Model discrimination ability
4. **Feature Importance**: Top contributing features

## Model Performance (Example Results)

```
Classification Report:
              precision    recall  f1-score   support
    No Flood       0.88      1.00      0.94       310
       Flood       1.00      0.95      0.98       910

ROC-AUC Score: 0.9931

Top Features:
1. T1d (Most recent precipitation)
2. Longitude
3. T2d
4. T10d
5. API_3day
```

## PINN Integration

### Threshold Logic
```python
if flood_probability > 0.7:
    trigger_stage2_pinn_simulation()
```

### Stage 1 Output
- **flood_probability**: Float [0.0, 1.0]
- **prediction**: Binary (Flood / No Flood)
- **trigger_pinn_stage2**: Boolean
- **risk_level**: CRITICAL / HIGH / MODERATE / LOW
- **confidence**: HIGH / MEDIUM

### Stage 2 Input (PINN)
When triggered (probability > 0.7):
- Station coordinates (Latitude, Longitude)
- Catchment characteristics
- Current precipitation pattern
- Predicted flood probability

### Stage 2 Output (PINN - Future Implementation)
- Detailed flood extent map
- Water depth distribution
- Flow velocity fields
- Optimal parameter estimates
- Time-series flood evolution

## Model Usage

### Training
```python
from flood_prediction_model import FloodPredictionModel

model = FloodPredictionModel(data_path='DATA/')
model.load_data()
      .prepare_flood_events()
      .generate_negative_samples(samples_per_station=10)
      .create_features()
      .train_model()
      .evaluate_model()
      .save_model()
```

### Prediction
```python
import pickle

# Load model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Prepare features
features = {
    'T1d': 85.5,
    'T2d': 120.3,
    'T3d': 95.8,
    # ... all 26 features
}

# Predict
result = predict_flood_risk(model_data, features)
print(f"Probability: {result['flood_probability']}")
print(f"Trigger PINN: {result['trigger_pinn_stage2']}")
```

## Limitations and Future Work

### Current Limitations
1. **Missing precipitation data** for negative samples (addressed via averaging)
2. **Static geomorphological features** (no temporal change consideration)
3. **Binary classification** (doesn't predict flood severity)
4. **Station-level predictions** (not spatial distribution)

### Future Enhancements
1. **Real-time data integration**: Weather forecasts, satellite imagery
2. **Temporal features**: Seasonal patterns, climate trends
3. **Multi-class classification**: Flood severity levels
4. **Spatial modeling**: Integrate with GIS data
5. **Deep learning**: LSTM for temporal sequences
6. **Ensemble methods**: Combine with XGBoost, LightGBM

## File Structure

```
PINN-Point-Climate/
├── DATA/                          # Input datasets
├── models/                        # Trained models
├── flood_prediction_model.py      # Main training script
├── example_prediction.py          # Demo script
├── requirements.txt               # Dependencies
├── README.md                      # User guide
├── TECHNICAL_DOCS.md              # This file
└── flood_model_evaluation.png    # Visualization output
```

## Dependencies

- pandas >= 1.5.0: Data manipulation
- numpy >= 1.23.0: Numerical operations
- scikit-learn >= 1.2.0: Machine learning
- matplotlib >= 3.6.0: Visualization
- seaborn >= 0.12.0: Statistical plots

## Performance Optimization

### Memory Usage
- Uses lazy loading where possible
- Categorical encoding reduces memory footprint
- Feature selection limits model complexity

### Training Time
- Random Forest: ~5-10 seconds on standard hardware
- Parallelization: n_jobs=-1 (uses all cores)
- Incremental training: Not implemented (not necessary for this dataset size)

### Inference Time
- Single prediction: <10ms
- Batch prediction (100 samples): <100ms
- Suitable for real-time applications

## Version History

- **v1.0** (Current): Initial implementation with Random Forest
  - Binary classification
  - 26 features
  - PINN integration threshold: 0.7

## Contact & Support

For questions or issues, please refer to the main README.md or contact the project maintainers.
