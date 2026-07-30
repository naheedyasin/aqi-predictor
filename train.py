# Trains Random Forest to predict PM2.5 at 24h/48h/72h ahead, for each city.
# Reads features from Hopsworks Feature Store (not local CSVs).
# Random Forest selected as final model after comparing against Ridge, Gradient Boosting,
# and a neural network (see git history for full comparison).

import joblib
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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

def train_and_evaluate(city, horizon):
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    df = fg.read()

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    
    target_column = f"target_pm25_{horizon}"
    
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

    persistence_predictions = X_test["pm2_5"]
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


if __name__ == "__main__":
    results = []
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("data", exist_ok=True)

    for city in CITIES:
        for horizon in HORIZONS:
            print(f"Training {city} / {horizon}...")
            result = train_and_evaluate(city, horizon)
            results.append(result)

            # Save model to local file first (Hopsworks needs a file path to upload)
            model_filename = f"saved_models/rf_{city}_{horizon}.pkl"
            joblib.dump(result["model"], model_filename)

            # Register with Hopsworks Model Registry
            model_registry_entry = mr.python.create_model(
                name=f"aqi_rf_{city}_{horizon}",
                metrics={
                    "test_r2": result["test_r2"],
                    "test_rmse": result["test_rmse"],
                    "test_mae": result["test_mae"],
                },
                description=f"Random Forest predicting PM2.5 {horizon} ahead for {city.capitalize()}"
            )
            model_registry_entry.save(model_filename)
            print(f"Registered model: aqi_rf_{city}_{horizon}")

    results_df = pd.DataFrame([{k: v for k, v in r.items() if k != "model"} for r in results])
    print("\n=== Final Results ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("data/model_results_summary.csv", index=False)
    print("\nSaved to data/model_results_summary.csv")