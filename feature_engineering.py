# Turns raw hourly AQI readings into a supervised learning dataset:
# adds time/lag/rolling features (the "situation") and target columns
# for AQI 24h/48h/72h later (the "answer" the model will learn to predict).

import pandas as pd
import numpy as np


CITIES = ["karachi", "lahore", "islamabad"]


def pm25_to_aqi(pm25):
    """US EPA breakpoint formula - converts raw PM2.5 (µg/m³) into the
    standard 0-500 AQI scale. Same formula used for display in app.py."""
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


def build_features(city):
    df = pd.read_csv(f"data/raw_historical_aqi_{city}.csv")

    # time-based features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # lag features (tells what was AQI/PM2.5 n hours ago.)
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["pm25_lag_1h"] = df["pm2_5"].shift(1)
    df["pm25_lag_3h"] = df["pm2_5"].shift(3)
    df["pm25_lag_24h"] = df["pm2_5"].shift(24)

    df["pm2_5_rolling_24h_mean"] = df["pm2_5"].rolling(window=24).mean()
    df["pm2_5_rolling_24h_std"] = df["pm2_5"].rolling(window=24).std()

    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1h"]

    # target columns = the actual answer the model tries to predict
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    df["target_pm25_24h"] = df["pm2_5"].shift(-24)
    df["target_pm25_48h"] = df["pm2_5"].shift(-48)
    df["target_pm25_72h"] = df["pm2_5"].shift(-72)

    # NEW: US EPA AQI targets, computed directly from future PM2.5 -
    # predicting these instead of raw PM2.5 avoids compounding error
    # through the conversion formula at inference time.
    df["target_aqi_us_24h"] = df["pm2_5"].shift(-24).apply(pm25_to_aqi)
    df["target_aqi_us_48h"] = df["pm2_5"].shift(-48).apply(pm25_to_aqi)
    df["target_aqi_us_72h"] = df["pm2_5"].shift(-72).apply(pm25_to_aqi)

    df = df.dropna()

    output_path = f"data/features_{city}.csv"
    df.to_csv(output_path, index=False)
    print(f"{city}: shape {df.shape}, saved to {output_path}")

    return df


if __name__ == "__main__":
    for city in CITIES:
        build_features(city)