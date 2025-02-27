import pandas as pd
import numpy as np
from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timedelta, timezone

# Function to generate synthetic solar data
def generate_solar_data(num_entries=50000):
    data = []
    base_date = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)  # Start date in UTC

    for _ in range(num_entries):
        # Random realistic latitude and longitude for solar applications (between -60 and 60 degrees)
        lat = np.random.uniform(-60, 60)
        lon = np.random.uniform(-180, 180)

        # Random date and time within 2023
        random_days = np.random.randint(0, 365)
        random_hours = np.random.randint(0, 24)
        random_minutes = np.random.randint(0, 60)
        date = base_date + timedelta(days=random_days, hours=random_hours, minutes=random_minutes)

        # Compute solar angles using Pysolar (ensure UTC time is used)
        altitude = get_altitude(lat, lon, date)
        azimuth = get_azimuth(lat, lon, date)

        # Correct Zenith Angle Calculation
        zenith_angle = max(0, min(90, 90 - altitude))  # Clamp between 0° and 90°

        # Weather data (random values within realistic ranges)
        temperature = np.random.uniform(-10, 40)  # °C
        dew_point = np.clip(temperature - np.random.uniform(2, 10), -15, 25)  # °C
        relative_humidity = np.random.uniform(10, 90)  # %
        pressure = np.clip(1013 - (0.12 * lat), 950, 1050)  # hPa
        wind_speed = np.random.uniform(0, 20)  # m/s

        # Solar radiation data (random realistic values)
        clearsky_dhi = np.random.uniform(50, 300)  # W/m²
        clearsky_dni = np.random.uniform(200, 1000)  # W/m²
        clearsky_ghi = clearsky_dni * np.random.uniform(0.75, 0.95)  # W/m²
        dhi = np.clip(clearsky_dhi * np.random.uniform(0.8, 1.0), 0, 300)
        dni = np.clip(clearsky_dni * np.random.uniform(0.8, 1.0), 0, 1000)
        ghi = np.clip(clearsky_ghi * np.random.uniform(0.8, 1.0), 0, 1200)

        # Surface albedo (random realistic values)
        surface_albedo = np.clip(np.random.uniform(0.1, 0.8), 0, 1)

        # Topocentric angles (azimuth adjustments)
        topo_zenith = zenith_angle
        topo_azimuth_east = azimuth
        topo_azimuth_west = (azimuth + 180) % 360  # Opposite azimuth direction

        # Julian day (day of the year)
        julian_day = date.timetuple().tm_yday

        # Append the entry to the dataset
        data.append([
            lat, lon, date, temperature, clearsky_dhi, clearsky_dni, clearsky_ghi,
            dew_point, dhi, dni, ghi, relative_humidity, zenith_angle, surface_albedo,
            pressure, wind_speed, topo_zenith, topo_azimuth_east, topo_azimuth_west, julian_day
        ])

    # Create DataFrame
    columns = [
        'Latitude', 'Longitude', 'Date', 'Temperature', 'Clearsky DHI', 'Clearsky DNI',
        'Clearsky GHI', 'Dew Point', 'DHI', 'DNI', 'GHI', 'Relative Humidity',
        'Solar Zenith Angle', 'Surface Albedo', 'Pressure', 'Wind Speed',
        'Topocentric Zenith Angle', 'Top. Azimuth Angle (Eastward from N)',
        'Top. Azimuth Angle (Westward from S)', 'Julian Day'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate dataset with 50,000 entries
dataset = generate_solar_data(num_entries=50000)

# Save to CSV
dataset.to_csv('solar_angle_dataset.csv', index=False)
print("Dataset generated and saved to 'solar_angle_dataset.csv'")
