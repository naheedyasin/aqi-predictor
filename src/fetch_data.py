import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# List of cities we're tracking — add/remove entries here, nothing else needs to change
CITIES = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
]

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

    row = {
        "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
        "city": city_name,
        "aqi": entry["main"]["aqi"],
        "co": entry["components"]["co"],
        "no": entry["components"]["no"],
        "no2": entry["components"]["no2"],
        "o3": entry["components"]["o3"],
        "so2": entry["components"]["so2"],
        "pm2_5": entry["components"]["pm2_5"],
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