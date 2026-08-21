# Runs hourly: fetches current AQI data for all cities, computes features
# using recent history from Hopsworks, and inserts the new row back in.

import os
import time
import tempfile
import requests
import pandas as pd
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import hopsworks
from hsfs.feature import Feature

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

# Timeout (seconds) for all outbound HTTP calls. Without this, a slow or
# hanging upstream API (this bit us with Open-Meteo) can stall the whole
# GitHub Actions run indefinitely instead of failing fast.
# Open-Meteo specifically has been observed timing out at 10s under load
# (this is what caused the Lahore failure), so it gets a longer budget.
REQUEST_TIMEOUT = 10
WEATHER_REQUEST_TIMEOUT = 20
FETCH_MAX_RETRIES = 3
FETCH_RETRY_BACKOFF_SECONDS = 5

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


def pm25_to_aqi(pm25):
    """Same EPA breakpoint formula as feature_engineering.py - keeps live
    hourly data consistent with the historical backfill's target columns."""
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


def fetch_with_retry(fetch_fn, label, city_name, max_retries=FETCH_MAX_RETRIES):
    """Generic retry+backoff wrapper for outbound API calls. Mirrors the
    pattern already used for Hopsworks read/insert - a single slow response
    from an external API shouldn't cost us the whole hourly row."""
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn()
        except requests.exceptions.RequestException as e:
            print(f"{label} attempt {attempt}/{max_retries} failed for {city_name}: {e}")
            if attempt == max_retries:
                return None
            time.sleep(FETCH_RETRY_BACKOFF_SECONDS * attempt)  # 5s, 10s, ...


def fetch_current(lat, lon):
    url = "http://api.openweathermap.org/data/2.5/air_pollution"
    params = {"lat": lat, "lon": lon, "appid": API_KEY}
    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.json()

def fetch_current_weather(lat, lon):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,wind_speed_10m,relative_humidity_2m,surface_pressure",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params, timeout=WEATHER_REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    current = data["current"]
    return {
        "temperature": current["temperature_2m"],
        "wind_speed": current["wind_speed_10m"],
        "humidity": current["relative_humidity_2m"],
        "pressure": current["surface_pressure"],
    }


def build_row(data, weather, city_name):
    """weather may be None if Open-Meteo failed after retries - we still
    write the pollution reading rather than lose the hour entirely.
    Missing weather fields become NaN, which downstream training already
    handles (see WEATHER_FEATURE_COLUMNS availability check)."""
    entry = data["list"][0]
    weather = weather or {}
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
        "temperature": weather.get("temperature"),
        "wind_speed": weather.get("wind_speed"),
        "humidity": weather.get("humidity"),
        "pressure": weather.get("pressure"),
    }


def read_with_retry(fg, city_name, max_retries=3):
    cutoff = datetime.now(timezone.utc) - timedelta(hours=72)
    cutoff_str = cutoff.strftime("%Y-%m-%d %H:%M:%S")
    for attempt in range(1, max_retries + 1):
        try:
            return fg.filter(Feature("timestamp") >= cutoff_str).read()
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

    raw_cols = ["timestamp", "city", "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3","temperature", "wind_speed", "humidity", "pressure"]
    recent_raw = recent_df[raw_cols].tail(48).copy()

    # Pollution data is the core signal - retry hard, and bail on this city
    # for this hour if it truly can't be fetched (better than inserting a
    # row with no AQI at all).
    raw_data = fetch_with_retry(
        lambda: fetch_current(city["lat"], city["lon"]), "OpenWeather pollution", city["name"]
    )
    if raw_data is None:
        print(f"Skipping {city['name']} - could not fetch pollution data after retries.")
        return

    # Weather is supplementary - retry, but don't lose the pollution
    # reading just because Open-Meteo is slow/down this hour.
    weather = fetch_with_retry(
        lambda: fetch_current_weather(city["lat"], city["lon"]), "Open-Meteo weather", city["name"]
    )
    if weather is None:
        print(f"Warning: {city['name']} - weather fetch failed after retries, "
              f"inserting row with weather fields as null.")

    new_row = build_row(raw_data, weather, city["name"])

    combined = pd.concat([recent_raw, pd.DataFrame([new_row])], ignore_index=True)
    combined = combined.drop_duplicates(subset="timestamp").sort_values("timestamp").reset_index(drop=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)

    # Reindex onto a fixed hourly grid so any missing hour becomes an
    # EXPLICIT NaN row instead of silently shifting every lag/target column
    # by one position. Without this, shift(24) means "24 rows back", which
    # only equals "24 hours back" if there are zero gaps - one retry
    # exhaustion (weather or pollution) anywhere in the last 72h would
    # otherwise corrupt every lag/target computed after that point.
    combined = combined.set_index("timestamp")
    full_hourly_index = pd.date_range(
        start=combined.index.min(), end=combined.index.max(), freq="h", tz="UTC"
    )
    n_missing = len(full_hourly_index) - len(combined)
    if n_missing > 0:
        print(f"{city['name']}: {n_missing} missing hour(s) in the last window - "
              f"reindexing so lag/target columns stay time-correct.")
    combined = combined.reindex(full_hourly_index)
    combined.index.name = "timestamp"
    combined = combined.reset_index()
    combined["city"] = city["name"].capitalize()  # restore - reindex NaNs it on gap rows

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

    combined["target_aqi_us_24h"] = combined["pm2_5"].shift(-24).apply(pm25_to_aqi)
    combined["target_aqi_us_48h"] = combined["pm2_5"].shift(-48).apply(pm25_to_aqi)
    combined["target_aqi_us_72h"] = combined["pm2_5"].shift(-72).apply(pm25_to_aqi)

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