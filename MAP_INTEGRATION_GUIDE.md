# 🗺️ Google Maps Integration Guide

## Overview

The web application allows you to click anywhere on a map to predict flood risk for that location using the trained ML model.

## Setup Instructions

### 1. Install Dependencies

```bash
pip install flask scipy
```

### 2. Get Google Maps API Key

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Create a new project or select existing one
3. Enable **Maps JavaScript API**
4. Create credentials (API Key)
5. Copy your API key

### 3. Configure API Key

Open `templates/map_interface.html` and replace `YOUR_API_KEY` with your actual API key:

```html
<!-- Line 397 - Replace YOUR_API_KEY -->
<script src="https://maps.googleapis.com/maps/api/js?key=YOUR_ACTUAL_API_KEY&callback=initMap" async defer></script>
```

**Example:**
```html
<script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyBXXXXXXXXXXXXXXXXXXXXX&callback=initMap" async defer></script>
```

### 4. Start the Application

```bash
python app.py
```

### 5. Open in Browser

Navigate to: **http://localhost:5000**

## How to Use

### Step 1: Select Location
- **Click anywhere on the map** to select coordinates
- OR **Click on existing station markers** (blue dots) to use known stations
- Coordinates will auto-fill in the sidebar

### Step 2: Enter Precipitation Data (Optional)
- Enter rainfall amounts (in mm) for the past 10 days
- T1d = Yesterday's rainfall
- T2d = 2 days ago, etc.
- If left empty, default moderate values will be used

### Step 3: Predict
- Click **"Predict Flood Risk"** button
- Wait for analysis (~1 second)

### Step 4: View Results
- **Flood Probability**: 0-100%
- **Risk Level**: LOW, MODERATE, HIGH, or CRITICAL
- **Nearest Station**: Closest monitoring station
- **Distance**: How far from your selected point
- **PINN Trigger**: Shows if Stage 2 simulation is recommended

## Features

### 🗺️ Interactive Map
- Pan and zoom across India
- Click anywhere to select coordinates
- View all 155 monitoring stations
- Station info windows with details

### 📊 Smart Prediction
- Uses **nearest station** data for predictions
- Calculates distance from your point to nearest station
- Applies station characteristics to your location

### 🎨 Visual Risk Indicators
- **Red (CRITICAL)**: Probability > 70%
- **Orange (HIGH)**: Probability 50-70%
- **Yellow (MODERATE)**: Probability 30-50%
- **Green (LOW)**: Probability < 30%

### ⚡ Real-time Analysis
- Instant predictions (<1 second)
- No page refresh needed
- Responsive interface

## API Endpoints

### GET `/`
Main map interface

### GET `/api/stations`
Returns all station locations as JSON
```json
[
  {
    "id": "INDOFLOODS-gauge-1013",
    "name": "Station Name",
    "lat": 28.5,
    "lon": 77.8,
    "river": "Ganga",
    "state": "Uttar Pradesh"
  }
]
```

### POST `/api/predict`
Predict flood risk for coordinates
```json
// Request
{
  "lat": 28.5,
  "lon": 77.8,
  "precipitation": {
    "T1d": 85.5,
    "T2d": 120.3,
    ...
  }
}

// Response
{
  "success": true,
  "prediction": {
    "probability": 0.804,
    "probability_percent": "80.4%",
    "prediction": "FLOOD",
    "risk_level": "CRITICAL",
    "risk_color": "#dc3545",
    "trigger_pinn": true,
    "confidence": "HIGH"
  },
  "location": {
    "clicked_lat": 28.5,
    "clicked_lon": 77.8,
    "nearest_station": "INDOFLOODS-gauge-1013",
    "station_name": "Station Name",
    "distance_km": "2.45",
    "river": "Ganga",
    "state": "Uttar Pradesh"
  }
}
```

## Example Scenarios

### Scenario 1: High Rainfall Area
1. Click on northern India (e.g., near Ganges basin)
2. Enter high precipitation values:
   - T1d: 120, T2d: 150, T3d: 100, etc.
3. Predict
4. **Result**: High probability, PINN triggered

### Scenario 2: Dry Season
1. Click anywhere
2. Enter low precipitation values:
   - T1d: 5, T2d: 8, T3d: 3, etc.
3. Predict
4. **Result**: Low probability, no PINN trigger

### Scenario 3: Using Existing Station
1. Click on blue station marker
2. Use default precipitation values
3. Predict
4. **Result**: Based on actual station characteristics

## Customization

### Change Default Precipitation
Edit `app.py` line 123:
```python
precipitation = {
    'T1d': 50.0,  # Change these values
    'T2d': 45.0,
    ...
}
```

### Change PINN Trigger Threshold
Edit `app.py` or modify model:
```python
probability_threshold = 0.7  # Change to 0.6, 0.8, etc.
```

### Modify Map Center/Zoom
Edit `templates/map_interface.html` line 285:
```javascript
map = new google.maps.Map(document.getElementById('map'), {
    center: { lat: 23.0, lng: 80.0 },  // Change coordinates
    zoom: 5,  // Change zoom level
    mapTypeId: 'terrain'  // or 'roadmap', 'satellite', 'hybrid'
});
```

## Troubleshooting

### Map not loading?
- Check if you replaced `YOUR_API_KEY` with actual key
- Verify Maps JavaScript API is enabled in Google Cloud Console
- Check browser console for errors (F12)

### "Station data not found" error?
- Ensure `DATA/metadata_indofloods.csv` exists
- Ensure `DATA/catchment_characteristics_indofloods.csv` exists
- Check file paths in `app.py`

### Predictions not working?
- Ensure model file exists: `models/flood_prediction_rf.pkl`
- Check if model was trained successfully
- Verify all CSV files are in `DATA/` folder

### Slow predictions?
- Model should predict in <10ms
- Check if you're using debug mode (it's slower)
- For production, set `debug=False` in `app.py`

## Production Deployment

For production use:

1. **Use production WSGI server**:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

2. **Restrict API key**:
   - In Google Cloud Console, restrict API key to your domain
   - Add HTTP referrer restrictions

3. **Add HTTPS**:
   - Use nginx or Apache as reverse proxy
   - Enable SSL/TLS

4. **Add rate limiting**:
```python
from flask_limiter import Limiter

limiter = Limiter(app, key_func=lambda: request.remote_addr)

@app.route('/api/predict', methods=['POST'])
@limiter.limit("10 per minute")
def predict():
    ...
```

## Future Enhancements

- [ ] Integrate real-time weather API (OpenWeatherMap, etc.)
- [ ] Add historical flood overlay on map
- [ ] Show catchment boundaries
- [ ] Export predictions to PDF/CSV
- [ ] Multi-location batch prediction
- [ ] Real-time monitoring dashboard
- [ ] SMS/Email alerts for high-risk areas

## Security Notes

⚠️ **Important**:
- Never commit API keys to Git
- Use environment variables for sensitive data
- Implement authentication for production
- Add CORS restrictions
- Validate all user inputs
- Sanitize coordinates before processing

## Support

For issues or questions:
1. Check TECHNICAL_DOCS.md for model details
2. Review Flask logs for errors
3. Check browser console for JavaScript errors
4. Verify all dependencies are installed

---

**Ready to predict floods on a map!** 🗺️🌊
