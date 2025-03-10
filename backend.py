from flask import Flask, render_template, request
import numpy as np
import tensorflow as tf
import joblib
from geopy.geocoders import Nominatim
import datetime
import math

# Load trained model & scaler
model = tf.keras.models.load_model('solar_angle_improved_model.h5')
scaler = joblib.load('scaler.pkl')

app = Flask(__name__)

def get_lat_lon(city_country):
    """Fetch latitude and longitude from city & country name."""
    geolocator = Nominatim(user_agent="solar_angle_app")
    try:
        location = geolocator.geocode(city_country, timeout=10)
        if location:
            return location.latitude, location.longitude
    except Exception as e:
        print(f"GeoPy error: {e}")
    return None, None

def compute_julian_day(year, month, day):
    """Compute the Julian day of the year."""
    return datetime.date(year, month, day).timetuple().tm_yday  

def estimate_missing_features(latitude, longitude, julian_day, hour):
    """Estimate missing features using empirical models & solar/weather approximations."""
    
    # Estimate solar position
    altitude = max(0, 90 - abs(latitude - (julian_day - 173) * 0.2))  
    zenith_angle = min(90, max(0, 90 - altitude))  # Ensure physical limits
    azimuth_angle = (hour / 24) * 360  

    # Approximate weather conditions
    temperature = np.clip(25 + (math.sin((julian_day / 365) * 2 * math.pi) * 10), -10, 40)
    dew_point = np.clip(temperature - np.random.uniform(5, 15), -15, 25)
    relative_humidity = np.random.uniform(10, 90)
    pressure = np.clip(1013 - (0.12 * latitude), 950, 1050)
    wind_speed = np.random.uniform(0, 20)

    # Approximate solar radiation
    clearsky_dhi = np.clip(np.random.uniform(50, 300), 0, 300)
    clearsky_dni = np.clip(np.random.uniform(200, 1000), 0, 1000)
    clearsky_ghi = np.clip(clearsky_dni * 0.85, 0, 1200)
    dhi = np.clip(clearsky_dhi * 0.9, 0, 300)
    dni = np.clip(clearsky_dni * 0.9, 0, 1000)
    ghi = np.clip(clearsky_ghi * 0.9, 0, 1200)

    return [
        temperature, clearsky_dhi, clearsky_dni, clearsky_ghi, dew_point,
        dhi, dni, ghi, relative_humidity, 0.2, pressure, wind_speed,
        zenith_angle, azimuth_angle, 360 - azimuth_angle, julian_day
    ]

def adjust_tilt_angle(raw_prediction):
    """Adjust tilt angle to be within the optimal range (0°-90°)."""
    if raw_prediction > 90:
        return 180 - raw_prediction  # Reflect to keep within range
    return raw_prediction

@app.route("/", methods=["GET", "POST"])
def index():
    predicted_tilt = None
    error = None

    if request.method == "POST":
        city_country = request.form.get("location")
        year = int(request.form.get("year"))
        month = int(request.form.get("month"))
        day = int(request.form.get("day"))
        hour = int(request.form.get("hour"))
        minute = int(request.form.get("minute"))  

        # Get latitude & longitude
        latitude, longitude = get_lat_lon(city_country)
        if latitude is None or longitude is None:
            error = "Could not find the location. Please enter a valid city and country."
        else:
            julian_day = compute_julian_day(year, month, day)
            estimated_features = estimate_missing_features(latitude, longitude, julian_day, hour)

            # Prepare input features
            input_features = np.array([[year, month, day, hour, minute, latitude, longitude] + estimated_features])
            input_features_scaled = scaler.transform(input_features.reshape(1, -1))  # Ensure correct shape
            
            # Model prediction
            raw_prediction = model.predict(input_features_scaled)[0][0]  
            predicted_tilt = round(adjust_tilt_angle(raw_prediction),2)

            # Debugging: Print the outputs
            print(f"Raw Prediction: {raw_prediction}, Adjusted Tilt Angle: {predicted_tilt}")

    return render_template("index.html", predicted_tilt=predicted_tilt, error=error)

if __name__ == "__main__":
    app.run(debug=True) 
