import os
import requests
import pandas as pd
from dotenv import load_dotenv
from datetime import datetime, timezone, timedelta

load_dotenv()

API_KEY = os.getenv("OPENWEATHER_API_KEY")

CITIES = [
    {"name": "Karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "Lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "Islamabad", "lat": 33.6844, "lon": 73.0479},
]


def pm25_to_aqi(pm25):
    """US EPA breakpoint formula - same formula used in feature_engineering.py
    and app.py. Used here so the raw backfill CSV stores AQI on the same
    0-500 EPA scale as everything downstream, instead of OpenWeather's own
    coarse 1-5 'main.aqi' index."""
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


def fetch_historical(lat, lon, start_date, end_date):
    url = "http://api.openweathermap.org/data/2.5/air_pollution/history"
    start_unix = int(start_date.timestamp())
    end_unix = int(end_date.timestamp())
    params = {
        "lat": lat,
        "lon": lon,
        "start": start_unix,
        "end": end_unix,
        "appid": API_KEY
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    data = response.json()
    return data


def parse_historical_to_dataframe(data, city_name):
    rows = []
    for entry in data["list"]:
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
        rows.append(row)
    return pd.DataFrame(rows)


def generate_month_ranges(months_back):
    ranges = []
    end = datetime.now(timezone.utc)
    for i in range(months_back):
        chunk_end = end - timedelta(days=30 * i)
        chunk_start = chunk_end - timedelta(days=30)
        ranges.append((chunk_start, chunk_end))
    return ranges


if __name__ == "__main__":
    ranges = generate_month_ranges(12)

    for city in CITIES:
        print(f"\n=== Backfilling {city['name']} ===")
        all_chunks = []

        for i, (start, end) in enumerate(ranges):
            print(f"Fetching month {i+1}/12: {start.date()} to {end.date()}")
            raw = fetch_historical(city["lat"], city["lon"], start, end)
            df_chunk = parse_historical_to_dataframe(raw, city["name"])
            all_chunks.append(df_chunk)

        full_df = pd.concat(all_chunks, ignore_index=True)
        full_df = full_df.drop_duplicates(subset="timestamp")
        full_df = full_df.sort_values("timestamp").reset_index(drop=True)

        filename = f"data/raw_historical_aqi_{city['name'].lower()}.csv"
        full_df.to_csv(filename, index=False)
        print(f"Saved {len(full_df)} rows to {filename}")