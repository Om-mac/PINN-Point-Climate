# 🌦️ Weather API Integration Setup

## Quick Setup (2 Options)

### Option 1: OpenWeatherMap (Recommended - Free Tier)

1. **Get API Key**: Go to https://openweathermap.org/api
   - Sign up for free account
   - Navigate to API Keys section
   - Copy your API key

2. **Set Environment Variable**:
   ```bash
   export OPENWEATHER_API_KEY='your_api_key_here'
   ```

3. **Check if Working**:
   ```bash
   python weather_api.py
   ```

### Option 2: Visual Crossing (Better Historical Data)

1. **Get API Key**: Go to https://www.visualcrossing.com/weather-api
   - Sign up for free account (1000 calls/day)
   - Get API key

2. **Set Environment Variable**:
   ```bash
   export VISUALCROSSING_API_KEY='your_api_key_here'
   ```

3. **Modify app.py** (line where fetch_real_precipitation is called):
   ```python
   precipitation = fetch_real_precipitation(lat, lon, api_provider='visualcrossing')
   ```

## Using the Feature

### In the Web Interface

1. Start the server: `python app.py`
2. Click on map location
3. **Check the box**: "📡 Fetch Real Weather Data (API)"
4. Click "Predict Flood Risk"
5. Results will show "Data Source: Real Weather API"

### Without API Key (Fallback)

If no API key is set:
- System automatically uses default moderate values (50mm, 45mm, etc.)
- Works perfectly fine for testing
- Shows "Data Source: Default Values"

## API Comparison

| Feature | OpenWeatherMap | Visual Crossing |
|---------|---------------|-----------------|
| **Free Tier** | 1000 calls/day | 1000 calls/day |
| **Historical Data** | Limited (forecast only) | ✅ Full 10 days |
| **Accuracy** | Good current weather | Excellent historical |
| **Speed** | Fast | Fast |
| **Recommended For** | Testing/Demo | Production |

## Testing Real Weather

```bash
# Test the weather API
python weather_api.py

# You'll see output like:
# Precipitation data for 28.6139, 77.209:
#   T1d: 45.2 mm
#   T2d: 38.7 mm
#   ...
```

## For Production

**Recommended**: Visual Crossing API
- More accurate historical precipitation
- Better for flood prediction
- Real 10-day historical data

**Setup**:
```bash
# 1. Get Visual Crossing API key
# 2. Set environment variable
export VISUALCROSSING_API_KEY='your_key'

# 3. Install requests
pip install requests

# 4. Restart Flask app
python app.py
```

## Cost Analysis

### Free Tier Limits
- **OpenWeatherMap**: 1000 calls/day = ~30,000/month
- **Visual Crossing**: 1000 calls/day = ~30,000/month

### For Heavy Usage
If you exceed free limits, consider:
- **Caching**: Store recent lookups (reduce API calls)
- **Paid Plans**: ~$40-100/month for unlimited
- **Alternative**: Use local weather station data

## Troubleshooting

**Issue**: "No API key found. Using default values."
- **Solution**: Set environment variable correctly
- **Check**: `echo $OPENWEATHER_API_KEY`

**Issue**: "Error fetching weather data"
- **Solution**: Check internet connection
- **Solution**: Verify API key is valid
- **Solution**: Check API call limits

**Issue**: Weather data seems inaccurate
- **Solution**: Switch to Visual Crossing (better historical)
- **Solution**: Verify coordinates are correct
- **Solution**: Check if location has weather stations nearby

## No API Key? No Problem!

The system works fine without API keys:
- Uses intelligent default values
- Based on moderate monsoon patterns
- Users can still enter manual values
- Perfect for testing and demos

---

**Ready to use real weather data!** 🌦️📡
