# Runs hourly: fetches current AQI data for all cities, computes features
# using recent history from Hopsworks, and inserts the new row back in.

import os
import json
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
REQUEST_TIMEOUT = 10
# Open-Meteo has been observed timing out at 10s under load (root cause of
# the Lahore failures) - give it more room.
WEATHER_REQUEST_TIMEOUT = 35
FETCH_MAX_RETRIES = 3

CITIES = [
    {"name": "karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "islamabad", "lat": 33.6844, "lon": 73.0479},
]

RAW_COLS = ["timestamp", "city", "aqi", "co", "no", "no2", "o3", "so2",
            "pm2_5", "pm10", "nh3", "temperature", "wind_speed", "humidity", "pressure"]

NUMERIC_COLS = ["aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
                 "temperature", "wind_speed", "humidity", "pressure"]

# Rows that failed to reach Hopsworks (pending) and a mirror of the last
# confirmed read (raw_cache) live here, keyed per city.
STATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "state")
os.makedirs(STATE_DIR, exist_ok=True)

temp_dir = tempfile.gettempdir()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)
fs = project.get_feature_store()

# Local state (pending buffer + raw cache)
def _state_path(city_name, kind):
    return os.path.join(STATE_DIR, f"{kind}_{city_name}.json")


def _load_state_df(city_name, kind):
    path = _state_path(city_name, kind)
    if not os.path.exists(path):
        return pd.DataFrame(columns=RAW_COLS)
    try:
        with open(path, "r") as f:
            records = json.load(f)
        if not records:
            return pd.DataFrame(columns=RAW_COLS)
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df
    except Exception as e:
        print(f"Could not read {kind} state for {city_name}: {e}")
        return pd.DataFrame(columns=RAW_COLS)


def _save_state_df(city_name, kind, df):
    path = _state_path(city_name, kind)
    if df is None or df.empty:
        if os.path.exists(path):
            os.remove(path)
        return
    out = df[RAW_COLS].copy()
    out["timestamp"] = out["timestamp"].astype(str)
    with open(path, "w") as f:
        json.dump(out.to_dict(orient="records"), f, default=str)


def _coerce_numeric_dtypes(df):
    """Hopsworks feature groups have a fixed schema (numeric features are
    typically float64/double). A single-row DataFrame containing a None
    becomes dtype 'object' rather than float64, and inserting an object
    column against a double feature is a common, silent cause of insert
    failures - this is what was almost certainly happening whenever the
    Open-Meteo call failed. Casting explicitly avoids that regardless of
    which columns had gaps."""
    df = df.copy()
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
    return df


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
    for attempt in range(1, max_retries + 1):
        try:
            return fetch_fn()
        except requests.exceptions.RequestException as e:
            print(f"{label} attempt {attempt}/{max_retries} failed for {city_name}: {e}")
            if attempt == max_retries:
                return None
            time.sleep(5 * attempt)  # 5s, 10s, ...


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


def build_row(data, weather, fallback_weather, city_name):
    """fallback_weather: dict of last-known weather values (carried forward
    from local history) used only when the live Open-Meteo call fails, so
    we never insert a None that breaks Hopsworks' typed schema, and never
    silently drop an hour's pollution reading just because the weather
    call flaked."""
    entry = data["list"][0]
    pm25 = entry["components"]["pm2_5"]
    weather = weather or fallback_weather or {}
    return {
        "timestamp": datetime.fromtimestamp(entry["dt"], tz=timezone.utc),
        "city": city_name.capitalize(),
        "aqi": pm25_to_aqi(pm25),
        "co": entry["components"]["co"],
        "no": entry["components"]["no"],
        "no2": entry["components"]["no2"],
        "o3": entry["components"]["o3"],
        "so2": entry["components"]["so2"],
        "pm2_5": pm25,
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


def insert_with_retry(fg, rows_df, city_name, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            fg.insert(rows_df)
            return True
        except Exception as e:
            print(f"Insert attempt {attempt}/{max_retries} failed for {city_name}: {e}")
            if attempt == max_retries:
                return False
            time.sleep(10)


def process_city(city):
    name = city["name"]
    print(f"Processing {name}...")

    fg = fs.get_feature_group(name=f"aqi_features_{name}", version=1)

    recent_df = read_with_retry(fg, name)

    if recent_df is not None:
        recent_df["timestamp"] = pd.to_datetime(recent_df["timestamp"], utc=True)
        recent_df = recent_df.sort_values("timestamp").reset_index(drop=True)
        confirmed_raw = recent_df[RAW_COLS].tail(72).copy()
        # Read succeeded - refresh the local mirror so a future transient
        # read failure still has recent history to fall back on.
        _save_state_df(name, "raw_cache", confirmed_raw)
    else:
        print(f"{name}: Hopsworks read failed after retries - falling back "
              f"to local raw cache instead of skipping the city entirely.")
        confirmed_raw = _load_state_df(name, "raw_cache")
        if confirmed_raw.empty:
            print(f"Skipping {name} - no Hopsworks read and no local cache available.")
            return

    # Rows fetched in previous runs that never made it into Hopsworks.
    pending_raw = _load_state_df(name, "pending")
    if not pending_raw.empty:
        print(f"{name}: retrying {len(pending_raw)} previously unconfirmed row(s).")

    raw_data = fetch_with_retry(
        lambda: fetch_current(city["lat"], city["lon"]), "OpenWeather pollution", name
    )
    if raw_data is None:
        print(f"Skipping {name} - could not fetch pollution data after retries.")
        return

    weather = fetch_with_retry(
        lambda: fetch_current_weather(city["lat"], city["lon"]), "Open-Meteo weather", name
    )

    fallback_weather = None
    if weather is None:
        fallback_source = pending_raw if not pending_raw.empty else confirmed_raw
        if not fallback_source.empty:
            last = fallback_source.sort_values("timestamp").iloc[-1]
            fallback_weather = {
                "temperature": last.get("temperature"),
                "wind_speed": last.get("wind_speed"),
                "humidity": last.get("humidity"),
                "pressure": last.get("pressure"),
            }
        print(f"{name}: Open-Meteo failed - carrying forward last known "
              f"weather instead of inserting nulls.")

    new_row = build_row(raw_data, weather, fallback_weather, name)
    new_row_df = pd.DataFrame([new_row])

    # Everything still needing to land in Hopsworks: old backlog + this hour.
    unconfirmed = pd.concat([pending_raw, new_row_df], ignore_index=True)
    unconfirmed["timestamp"] = pd.to_datetime(unconfirmed["timestamp"], utc=True)
    unconfirmed = unconfirmed.drop_duplicates(subset="timestamp", keep="last")

    combined = pd.concat([confirmed_raw, unconfirmed], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"], utc=True)
    combined = combined.drop_duplicates(subset="timestamp", keep="last") \
                        .sort_values("timestamp").reset_index(drop=True)
    combined = _coerce_numeric_dtypes(combined)

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

    # Only the rows still unconfirmed need to be (re)inserted - already
    # confirmed rows are left alone. They're recomputed against the full
    # combined history above, so lag/rolling values are correct even after
    # a multi-hour gap.
    rows_to_insert = combined[combined["timestamp"].isin(unconfirmed["timestamp"])].copy()

    success = insert_with_retry(fg, rows_to_insert, name)
    if success:
        print(f"Inserted {len(rows_to_insert)} row(s) for {name}, "
              f"latest at {rows_to_insert['timestamp'].max()}")
        _save_state_df(name, "pending", pd.DataFrame(columns=RAW_COLS))
    else:
        print(f"{name}: insert failed after retries - buffering "
              f"{len(unconfirmed)} row(s) locally to retry next run.")
        _save_state_df(name, "pending", unconfirmed)


if __name__ == "__main__":
    for city in CITIES:
        try:
            process_city(city)
        except Exception as e:
            print(f"Unexpected error processing {city['name']}: {e}")
            continue