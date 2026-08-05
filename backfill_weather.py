# Backfills historical weather (temperature, wind, humidity, pressure) using
# Open-Meteo's free Archive API - no key needed, no rate limits like
# OpenWeather's paid historical tier would require.

import requests
import pandas as pd
from datetime import datetime, timezone, timedelta

CITIES = [
    {"name": "karachi", "lat": 24.8607, "lon": 67.0011},
    {"name": "lahore", "lat": 31.5497, "lon": 74.3436},
    {"name": "islamabad", "lat": 33.6844, "lon": 73.0479},
]


def fetch_historical_weather(lat, lon, start_date, end_date):
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,wind_speed_10m,relative_humidity_2m,surface_pressure",
        "timezone": "UTC",
    }
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()


def parse_weather_to_dataframe(data, city_name):
    hourly = data["hourly"]
    df = pd.DataFrame({
        "timestamp": pd.to_datetime(hourly["time"], utc=True),
        "city": city_name.capitalize(),
        "temperature": hourly["temperature_2m"],
        "wind_speed": hourly["wind_speed_10m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["surface_pressure"],
    })
    return df


if __name__ == "__main__":
    end_date = datetime.now(timezone.utc)
    start_date = end_date - timedelta(days=365)

    for city in CITIES:
        print(f"Fetching historical weather for {city['name']}...")
        data = fetch_historical_weather(city["lat"], city["lon"], start_date, end_date)
        df = parse_weather_to_dataframe(data, city["name"])

        output_path = f"data/raw_historical_weather_{city['name']}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}")