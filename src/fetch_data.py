import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
]


def pm25_to_aqi(pm25):
    """US EPA breakpoint formula - same formula used in feature_engineering.py,
    backfill_data.py, hourly_pipeline.py, and app.py. Keeps this script's
    'aqi' column on the same 0-500 scale as everywhere else in the project,
    instead of OpenWeather's own coarse 1-5 'main.aqi' index."""
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ]
    if pd.isna(pm25):
        return None
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
    return 500


def fetch_air_pollution(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {
        "lat": lat,
        "lon": lon,
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data

def parse_to_dataframe(data, city_name):
    entry = data["list"][0]
    pm25 = entry["components"]["pm2_5"]

    row = {
        "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
        "city": city_name,
        "aqi": pm25_to_aqi(pm25),
        "co": entry["components"]["co"],
        "no": entry["components"]["no"],
        "no2": entry["components"]["no2"],
        "o3": entry["components"]["o3"],
        "so2": entry["components"]["so2"],
        "pm2_5": pm25,
        "pm10": entry["components"]["pm10"],
        "nh3": entry["components"]["nh3"],
    }
    return pd.DataFrame([row])

if __name__ == "__main__":
    all_rows = []
    for city in CITIES:
        print(f"Fetching current data for {city['name']}...")
        raw_data = fetch_air_pollution(city["lat"], city["lon"])
        df = parse_to_dataframe(raw_data, city["name"])
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True)
    print(combined)