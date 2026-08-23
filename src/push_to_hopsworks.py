import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

temp_dir = tempfile.gettempdir()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)

fs = project.get_feature_store()
print(f"Connected to feature store: {fs.name}")

CITIES = ["karachi", "lahore", "islamabad"]

def push_city_features(city):
    df = pd.read_csv(f"data/features_{city}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    weather_cols = ["temperature", "wind_speed", "humidity", "pressure"]
    df[weather_cols] = df[weather_cols].astype(float)

    fg = fs.get_or_create_feature_group(
        name=f"aqi_features_{city}",
        version=1,
        description=f"Hourly AQI/pollutant features for {city.capitalize()}, with lag/rolling features and PM2.5 targets",
        primary_key=["timestamp"],
        event_time="timestamp",
        time_travel_format="HUDI",
        online_enabled=False
    )

    fg.insert(df)
    print(f"Inserted {len(df)} rows into aqi_features_{city}")


if __name__ == "__main__":
    for city in CITIES:
        push_city_features(city)