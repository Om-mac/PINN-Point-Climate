# 🚀 Quick Start Guide - PINN-Point Climate Stage 1

Get started with the flood prediction model in 5 minutes!

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation (1 minute)

```bash
# Navigate to project directory
cd PINN-Point-Climate

# Install dependencies
pip install -r requirements.txt
```

## Train Model (2 minutes)

```bash
# Train the model with default settings
python flood_prediction_model.py
```

**What it does:**
- ✅ Loads 4 CSV files from DATA/ directory
- ✅ Generates 1,550 negative samples (non-flood dates)
- ✅ Creates 26 features including API (3, 7, 10 day)
- ✅ Trains Random Forest with 4,878 training samples
- ✅ Evaluates on 1,220 test samples
- ✅ Saves model to `models/flood_prediction_rf.pkl`
- ✅ Generates visualization: `flood_model_evaluation.png`

**Expected Output:**
```
ROC-AUC Score: 0.99+
Accuracy: 97%+
Samples triggering PINN Stage 2: ~64%
```

## Run Example Predictions (1 minute)

```bash
# Run demo predictions
python example_prediction.py
```

**What it shows:**
- Example 1: High-risk scenario (heavy rainfall)
- Example 2: Low-risk scenario (minimal rainfall)
- PINN trigger logic demonstration

## Make Your Own Predictions (1 minute)

Create a new file `my_prediction.py`:

```python
import pickle
import pandas as pd

# Load trained model
with open('models/flood_prediction_rf.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
feature_columns = model_data['feature_columns']

# Your station data
my_features = {
    # Recent precipitation (mm)
    'T1d': 85.5,      # Yesterday
    'T2d': 120.3,     # 2 days ago
    'T3d': 95.8,      # 3 days ago
    'T4d': 110.2,
    'T5d': 98.4,
    'T6d': 105.6,
    'T7d': 88.9,
    'T8d': 92.1,
    'T9d': 78.3,
    'T10d': 82.7,
    
    # API (calculate or provide)
    'API_3day': 301.6,
    'API_7day': 704.7,
    'API_10day': 957.8,
    
    # Station characteristics
    'Drainage Area': 3542.1,
    'Catchment Relief': 4416.0,
    'Catchment Length': 116870.2,
    'Relief Ratio': 0.0378,
    'Drainage Density': 0.000192,
    'Stream Order': 4.0,
    'Catchment Area': 3542.1,
    'Latitude': 28.5,
    'Longitude': 77.8,
    
    # Encoded categories (0, 1, 2, ...)
    'Land cover_encoded': 1,
    'Soil type_encoded': 0,
    'lithology type_encoded': 2,
    'KoppenGeiger Climate Type_encoded': 1
}

# Ensure all features are present
feature_dict = {col: my_features.get(col, 0.0) for col in feature_columns}
feature_df = pd.DataFrame([feature_dict])

# Predict
probability = model.predict_proba(feature_df)[0, 1]

print(f"Flood Probability: {probability:.1%}")
print(f"Trigger PINN Stage 2: {'YES ⚠️' if probability > 0.7 else 'NO'}")
```

Run it:
```bash
python my_prediction.py
```

## Understanding the Output

### Probability Ranges
| Probability | Risk Level | PINN Stage 2 | Action |
|------------|------------|--------------|---------|
| 0.0 - 0.3  | LOW        | ❌ No        | Monitor |
| 0.3 - 0.5  | MODERATE   | ❌ No        | Watch   |
| 0.5 - 0.7  | HIGH       | ❌ No        | Alert   |
| 0.7 - 1.0  | CRITICAL   | ✅ Yes       | **Trigger PINN** |

### PINN Integration
When probability > 0.7:
```
Stage 1 (RF Model) → Probability: 0.82
                     ↓
              Trigger PINN Stage 2
                     ↓
         Detailed Hydraulic Simulation
         (Flood extent, depth, velocity)
```

## Visualizations

After training, check `flood_model_evaluation.png` for:
1. **Confusion Matrix**: Model accuracy breakdown
2. **ROC Curve**: Discrimination ability
3. **Feature Importance**: Top contributing factors
4. **Probability Distribution**: Risk stratification

## Common Issues

### Issue 1: Missing CSV files
```
FileNotFoundError: DATA/floodevents_indofloods.csv
```
**Solution**: Ensure all 4 CSV files are in `DATA/` directory

### Issue 2: Module not found
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution**: `pip install -r requirements.txt`

### Issue 3: Low model performance
**Solution**: Check data quality, try adjusting:
- `samples_per_station` (default: 10)
- `test_size` (default: 0.2)
- Random Forest hyperparameters in code

## Next Steps

1. **Explore the data**: Open CSV files in pandas
2. **Tune hyperparameters**: Modify Random Forest settings
3. **Try XGBoost**: Alternative to Random Forest (mentioned in goal)
4. **Implement Stage 2**: Build PINN for detailed simulation

## File Locations

| File | Purpose |
|------|---------|
| `DATA/*.csv` | Input datasets |
| `models/flood_prediction_rf.pkl` | Trained model |
| `flood_model_evaluation.png` | Visualizations |
| `flood_prediction_model.py` | Training script |
| `example_prediction.py` | Demo script |

## Need Help?

1. Check `README.md` for detailed documentation
2. Read `TECHNICAL_DOCS.md` for technical details
3. Examine code comments in `.py` files
4. Review example outputs

## Performance Benchmarks

On standard laptop (8GB RAM, i5 processor):
- Data loading: ~1 second
- Feature engineering: ~2 seconds
- Model training: ~5 seconds
- Evaluation: ~1 second
- Total time: **~10 seconds**

## Quick Checklist

- [ ] Python 3.8+ installed
- [ ] All dependencies installed
- [ ] DATA/ folder with 4 CSV files
- [ ] Ran `flood_prediction_model.py` successfully
- [ ] Model saved to `models/` directory
- [ ] Visualization generated
- [ ] Tested `example_prediction.py`
- [ ] Made custom prediction

✅ **You're ready to integrate with PINN Stage 2!**

---

**Questions?** Check the main README.md or TECHNICAL_DOCS.md for more details.
