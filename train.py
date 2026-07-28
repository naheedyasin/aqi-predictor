# Trains Random Forest to predict PM2.5 24h ahead (Karachi baseline).
# Compared against Ridge and naive persistence; RF won (Test R² 0.44 vs 0.23/0.17).
# Tuned regularization to reduce overfitting (train/test gap: 0.53 → 0.37).

import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

df = pd.read_csv("data/features_karachi.csv")
df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

split_index = int(len(df) * 0.8)
train_df = df.iloc[:split_index]
test_df = df.iloc[split_index:]

print(f"Train: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"Test: {test_df['timestamp'].min()} to {test_df['timestamp'].max()}")

feature_columns = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm2_5_rolling_24h_mean",
    "aqi_change_rate"
]
target_column = "target_pm25_24h"

X_train = train_df[feature_columns]
y_train = train_df[target_column]

X_test = test_df[feature_columns]
y_test = test_df[target_column]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test: {X_test.shape}, y_test: {y_test.shape}")

model = Ridge()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

rmse = mean_squared_error(y_test, predictions) ** 0.5
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"RMSE: {rmse:.3f}")
print(f"MAE: {mae:.3f}")
print(f"R²: {r2:.3f}")

persistence_predictions = X_test["pm2_5"]  # naive guess: "PM2.5 stays the same as right now"

persistence_rmse = mean_squared_error(y_test, persistence_predictions) ** 0.5
persistence_mae = mean_absolute_error(y_test, persistence_predictions)
persistence_r2 = r2_score(y_test, persistence_predictions)

print(f"\n--- Persistence baseline (naive: tomorrow = today) ---")
print(f"RMSE: {persistence_rmse:.3f}")
print(f"MAE: {persistence_mae:.3f}")
print(f"R²: {persistence_r2:.3f}")

rf_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=20,
    max_features="sqrt",
    random_state=42
)
rf_model.fit(X_train, y_train)

# check training performance first (helps us see if the model is overfitting)
rf_train_predictions = rf_model.predict(X_train)
rf_train_r2 = r2_score(y_train, rf_train_predictions)

# now check test performance (the real, honest measure of how good this model is)
rf_predictions = rf_model.predict(X_test)
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print(f"\n--- Random Forest ---")
print(f"Train R²: {rf_train_r2:.3f}")
print(f"Test RMSE: {rf_rmse:.3f}")
print(f"Test MAE: {rf_mae:.3f}")
print(f"Test R²: {rf_r2:.3f}")