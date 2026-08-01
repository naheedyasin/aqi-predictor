# Runs hourly: fetches current AQI data for all cities, computes features
# using recent history from Hopsworks, and inserts the new row back in.

import os
import time
import tempfile
import requests
import pandas as pd
from datetime import datetime, timezone
from dotenv import load_dotenv
import hopsworks

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = [
    {"name": "karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "islamabad", "lat": 33.6844, "lon": 73.0479},
]

temp_dir = tempfile.gettempdir()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)
fs = project.get_feature_store()


def fetch_current(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def build_row(data, city_name):
    entry = data["list"][0]
    return {
        "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
        "city": city_name.capitalize(),
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


def read_with_retry(fg, city_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            return fg.read()
        except Exception as e:
            print(f"Read attempt {attempt}/{max_retries} failed for {city_name}: {e}")
            if attempt == max_retries:
                return None
            time.sleep(10)


def insert_with_retry(fg, row, city_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(row)
            return True
        except Exception as e:
            print(f"Insert attempt {attempt}/{max_retries} failed for {city_name}: {e}")
            if attempt == max_retries:
                return False
            time.sleep(10)


def process_city(city):
    print(f"Processing {city['name']}...")

    fg = fs.get_feature_group(name=f"aqi_features_{city['name']}", version=1)

    recent_df = read_with_retry(fg, city["name"])
    if recent_df is None:
        print(f"Skipping {city['name']} - could not read data after retries.")
        return

    recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"])
    recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)

    raw_cols = ["timestamp", "city", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    recent_raw = recent_df[raw_cols].tail(48).copy()

    raw_data = fetch_current(city["lat"], city["lon"])
    new_row = build_row(raw_data, city["name"])

    combined = pd.concat([recent_raw, pd.DataFrame([new_row])], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

    combined["hour"] = combined["timestamp"].dt.hour.astype("int64")
    combined["day_of_week"] = combined["timestamp"].dt.dayofweek.astype("int64")
    combined["month"] = combined["timestamp"].dt.month.astype("int64")
    combined["is_weekend"] = combined["day_of_week"].isin([5, 6]).astype("int64")

    combined["aqi_lag_1h"] = combined["aqi"].shift(1)
    combined["aqi_lag_3h"] = combined["aqi"].shift(3)
    combined["aqi_lag_24h"] = combined["aqi"].shift(24)

    combined["pm25_lag_1h"] = combined["pm2_5"].shift(1)
    combined["pm25_lag_3h"] = combined["pm2_5"].shift(3)
    combined["pm25_lag_24h"] = combined["pm2_5"].shift(24)

    combined["pm2_5_rolling_24h_mean"] = combined["pm2_5"].rolling(window=24).mean()
    combined["pm2_5_rolling_24h_std"] = combined["pm2_5"].rolling(window=24).std()

    combined["aqi_change_rate"] = combined["aqi"] - combined["aqi_lag_1h"]

    combined["target_aqi_24h"] = combined["aqi"].shift(-24)
    combined["target_aqi_48h"] = combined["aqi"].shift(-48)
    combined["target_aqi_72h"] = combined["aqi"].shift(-72)
    combined["target_pm25_24h"] = combined["pm2_5"].shift(-24)
    combined["target_pm25_48h"] = combined["pm2_5"].shift(-48)
    combined["target_pm25_72h"] = combined["pm2_5"].shift(-72)

    latest_row = combined.tail(1)

    success = insert_with_retry(fg, latest_row, city["name"])
    if success:
        print(f"Inserted new row for {city['name']} at {latest_row['timestamp'].values[0]}")
    else:
        print(f"Skipping {city['name']} - could not insert after retries.")


if __name__ == "__main__":
    for city in CITIES:
        try:
            process_city(city)
        except Exception as e:
            print(f"Unexpected error processing {city['name']}: {e}")
            continue