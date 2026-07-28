# Trains Random Forest to predict PM2.5 at 24h/48h/72h ahead, for each city.
# Configuration proven on Karachi/24h baseline (RF beat Ridge and naive persistence).
# Loops the same logic across all cities and horizons, saving metrics for comparison.

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

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
    df = pd.read_csv(f"data/features_{city}.csv")
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

    return {
        "city": city,
        "horizon": horizon,
        "train_r2": round(train_r2, 3),
        "test_rmse": round(rmse, 3),
        "test_mae": round(mae, 3),
        "test_r2": round(test_r2, 3),
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