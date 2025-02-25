import pandas as pd
import numpy as np
from pysolar.solar import get_altitude, get_azimuth
from datetime import datetime, timedelta, timezone

# Function to generate synthetic data
def generate_synthetic_data(num_entries=50000):
    data = []
    base_date = datetime(2023, 1, 1, 0, 0, tzinfo=timezone.utc)  # Start date

    for i in range(num_entries):
        # Random latitude and longitude
        lat = np.random.uniform(-90, 90)
        lon = np.random.uniform(-180, 180)

        # Random date and time within 2023
        random_days = np.random.randint(0, 365)
        random_hours = np.random.randint(0, 24)
        date = base_date + timedelta(days=random_days, hours=random_hours)

        # Solar angles
        altitude = get_altitude(lat, lon, date)
        azimuth = get_azimuth(lat, lon, date)
        zenith_angle = 90 - altitude

        # Weather data (random values within realistic ranges)
        temperature = np.random.uniform(-10, 40)  # Temperature in °C
        dew_point = np.random.uniform(-15, 25)    # Dew point in °C
        relative_humidity = np.random.uniform(0, 100)  # Relative humidity in %
        pressure = np.random.uniform(950, 1050)  # Atmospheric pressure in hPa
        wind_speed = np.random.uniform(0, 25)    # Wind speed in m/s

        # Solar radiation data (random values within realistic ranges)
        clearsky_dhi = np.random.uniform(0, 300)  # Clearsky DHI in W/m²
        clearsky_dni = np.random.uniform(0, 1000)  # Clearsky DNI in W/m²
        clearsky_ghi = np.random.uniform(0, 1200)  # Clearsky GHI in W/m²
        dhi = np.random.uniform(0, 300)  # DHI in W/m²
        dni = np.random.uniform(0, 1000)  # DNI in W/m²
        ghi = np.random.uniform(0, 1200)  # GHI in W/m²

        # Surface albedo (random value between 0 and 1)
        surface_albedo = np.random.uniform(0, 1)

        # Topocentric angles (approximated as solar angles)
        topo_zenith = zenith_angle
        topo_azimuth_east = azimuth
        topo_azimuth_west = 360 - azimuth

        # Julian day (day of the year)
        julian_day = date.timetuple().tm_yday

        # Append the entry to the dataset
        data.append([
            lat, lon, date, temperature, clearsky_dhi, clearsky_dni, clearsky_ghi,
            dew_point, dhi, dni, ghi, relative_humidity, zenith_angle, surface_albedo,
            pressure, wind_speed, topo_zenith, topo_azimuth_east, topo_azimuth_west, julian_day
        ])

    # Create a DataFrame
    columns = [
        'Latitude', 'Longitude', 'Date', 'Temperature', 'Clearsky DHI', 'Clearsky DNI',
        'Clearsky GHI', 'Dew Point', 'DHI', 'DNI', 'GHI', 'Relative Humidity',
        'Solar Zenith Angle', 'Surface Albedo', 'Pressure', 'Wind Speed',
        'Topocentric Zenith Angle', 'Top. Azimuth Angle (Eastward from N)',
        'Top. Azimuth Angle (Westward from S)', 'Julian Day'
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

# Generate 1000 entries
dataset = generate_synthetic_data(num_entries=1000)

# Save to CSV
dataset.to_csv('solar_angle_dataset.csv', index=False)
print("Dataset generated and saved to 'solar_angle_dataset.csv'")