# Trains Random Forest to predict AQI (US EPA scale) directly at 24h/48h/72h ahead, for each city.
# Reads features from Hopsworks Feature Store (not local CSVs).
# Switched from predicting PM2.5 (then converting) to predicting AQI directly,
# avoids compounding error through the conversion formula.

import joblib
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import time

load_dotenv()

temp_dir = tempfile.gettempdir()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)

fs = project.get_feature_store()
print(f"Connected to feature store: {fs.name}")

mr = project.get_model_registry()

CITIES = ["karachi", "lahore", "islamabad"]
HORIZONS = ["24h", "48h", "72h"]

feature_columns = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm2_5_rolling_24h_mean",
    "aqi_change_rate"
]


def pm25_to_aqi(pm25):
    """Same EPA breakpoint formula as feature_engineering.py - used here only
    to build the persistence baseline (naive: 'AQI stays the same as now')."""
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


def train_and_evaluate(city, horizon):
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    df = fg.read()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    target_column = f"target_aqi_us_{horizon}"

    df = df.dropna(subset=feature_columns + [target_column])

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=8,
        min_samples_leaf=20,
        max_features="sqrt",
        random_state=42
    )
    rf_model.fit(X_train, y_train)

    train_r2 = r2_score(y_train, rf_model.predict(X_train))

    predictions = rf_model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    mae = mean_absolute_error(y_test, predictions)
    test_r2 = r2_score(y_test, predictions)

    persistence_predictions = X_test["pm2_5"].apply(pm25_to_aqi)
    persistence_r2 = r2_score(y_test, persistence_predictions)

    return {
        "city": city,
        "horizon": horizon,
        "model": rf_model,
        "train_r2": round(train_r2, 3),
        "test_rmse": round(rmse, 3),
        "test_mae": round(mae, 3),
        "test_r2": round(test_r2, 3),
        "persistence_r2": round(persistence_r2, 3),
    }
    
def register_model_with_retry(mr, model_filename, name, metrics, description, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            model_registry_entry = mr.python.create_model(
                name=name,
                metrics=metrics,
                description=description
            )
            model_registry_entry.save(model_filename)
            return True
        except Exception as e:
            print(f"Registration attempt {attempt}/{max_retries} failed for {name}: {e}")
            if attempt == max_retries:
                print(f"Skipping registration for {name} after {max_retries} failed attempts.")
                return False
            time.sleep(15)


if __name__ == "__main__":
    results = []
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for city in CITIES:
        for horizon in HORIZONS:
            print(f"Training {city} / {horizon}...")
            result = train_and_evaluate(city, horizon)
            results.append(result)

            model_filename = f"saved_models/rf_{city}_{horizon}.pkl"
            joblib.dump(result["model"], model_filename)

            success = register_model_with_retry(
                mr, model_filename,
                name=f"aqi_rf_{city}_{horizon}",
                metrics={
                    "test_r2": result["test_r2"],
                    "test_rmse": result["test_rmse"],
                    "test_mae": result["test_mae"],
                },
                description=f"Random Forest predicting AQI (US EPA scale) {horizon} ahead for {city.capitalize()}"
            )
            if success:
                print(f"Registered model: aqi_rf_{city}_{horizon}")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    print("\n=== Final Results ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("data/model_results_summary.csv", index=False)
    print("\nSaved to data/model_results_summary.csv")