# Turns raw hourly AQI readings into a supervised learning dataset:
# adds time/lag/rolling features (the "situation") and target columns
# for AQI 24h/48h/72h later (the "answer" the model will learn to predict).

import pandas as pd
import numpy as np


CITIES = ["karachi", "lahore", "islamabad"]

def build_features(city):
    df = pd.read_csv(f"data/raw_historical_aqi_{city}.csv")

    # time-based features
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"] = df["timestamp"].dt.month
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # lag features (tells what was AQI n hours ago.)
    df["aqi_lag_1h"] = df["aqi"].shift(1)
    df["aqi_lag_3h"] = df["aqi"].shift(3)
    df["aqi_lag_24h"] = df["aqi"].shift(24)

    df["pm2_5_rolling_24h_mean"] = df["pm2_5"].rolling(window=24).mean()

    df["aqi_change_rate"] = df["aqi"] - df["aqi_lag_1h"]

    # target columns = the actual answer the model tries to predict
    df["target_aqi_24h"] = df["aqi"].shift(-24)
    df["target_aqi_48h"] = df["aqi"].shift(-48)
    df["target_aqi_72h"] = df["aqi"].shift(-72)

    df = df.dropna()

    output_path = f"data/features_{city}.csv"
    df.to_csv(output_path, index=False)
    print(f"{city}: shape {df.shape}, saved to {output_path}")

    return df


if __name__ == "__main__":
    for city in CITIES:
        build_features(city)
