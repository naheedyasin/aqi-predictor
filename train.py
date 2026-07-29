# Trains Random Forest to predict PM2.5 at 24h/48h/72h ahead, for each city.
# Reads features from Hopsworks Feature Store (not local CSVs).
# Configuration proven on Karachi/24h baseline (RF beat Ridge and naive persistence).

import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
import tensorflow as tf
from tensorflow import keras

load_dotenv()

temp_dir = tempfile.gettempdir()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)

fs = project.get_feature_store()
print(f"Connected to feature store: {fs.name}")

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

    split_index = int(len(df) * 0.8)
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    target_column = f"target_pm25_{horizon}"

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
    
    gb_model = GradientBoostingRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        min_samples_leaf=20,
        random_state=42
    )
    gb_model.fit(X_train, y_train)

    gb_train_r2 = r2_score(y_train, gb_model.predict(X_train))
    gb_predictions = gb_model.predict(X_test)
    gb_rmse = mean_squared_error(y_test, gb_predictions) ** 0.5
    gb_mae = mean_absolute_error(y_test, gb_predictions)
    gb_test_r2 = r2_score(y_test, gb_predictions)
    
    
    # Neural network needs scaled inputs (unlike tree-based models) for stable training
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    nn_model = keras.Sequential([
        keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1)
    ])

    nn_model.compile(optimizer="adam", loss="mse")
    nn_model.fit(X_train_scaled, y_train, epochs=30, batch_size=32, verbose=0)

    nn_train_r2 = r2_score(y_train, nn_model.predict(X_train_scaled, verbose=0).flatten())
    nn_predictions = nn_model.predict(X_test_scaled, verbose=0).flatten()
    nn_rmse = mean_squared_error(y_test, nn_predictions) ** 0.5
    nn_mae = mean_absolute_error(y_test, nn_predictions)
    nn_test_r2 = r2_score(y_test, nn_predictions)

    return {
        "city": city,
        "horizon": horizon,
        "rf_train_r2": round(train_r2, 3),
        "rf_test_rmse": round(rmse, 3),
        "rf_test_mae": round(mae, 3),
        "rf_test_r2": round(test_r2, 3),
        "gb_train_r2": round(gb_train_r2, 3),
        "gb_test_rmse": round(gb_rmse, 3),
        "gb_test_mae": round(gb_mae, 3),
        "gb_test_r2": round(gb_test_r2, 3),
        "nn_train_r2": round(nn_train_r2, 3),
        "nn_test_rmse": round(nn_rmse, 3),
        "nn_test_mae": round(nn_mae, 3),
        "nn_test_r2": round(nn_test_r2, 3),
        "persistence_r2": round(persistence_r2, 3),
    }


if __name__ == "__main__":
    results = []
    for city in CITIES:
        for horizon in HORIZONS:
            print(f"Training {city} / {horizon}...")
            result = train_and_evaluate(city, horizon)
            results.append(result)

    results_df = pd.DataFrame(results)
    print("\n=== Final Results ===")
    print(results_df.to_string(index=False))

    results_df.to_csv("data/model_results_summary.csv", index=False)
    print("\nSaved to data/model_results_summary.csv")