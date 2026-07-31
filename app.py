import streamlit as st
import os
import tempfile
import pandas as pd
from dotenv import load_dotenv
import hopsworks

load_dotenv()

st.title("AQI Predictor")

temp_dir = tempfile.gettempdir()

@st.cache_resource
def get_hopsworks_project():
    project = hopsworks.login(
        api_key_value=os.getenv("HOPSWORKS_API_KEY"),
        cert_folder=temp_dir
    )
    return project

project = get_hopsworks_project()
fs = project.get_feature_store()

st.write("Connected to Hopsworks!")

CITY = "karachi"

@st.cache_data(ttl=300)
def load_features(city):
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

df = load_features(CITY)

st.write(f"Latest data for {CITY.capitalize()}:")
st.dataframe(df.tail(5))

import joblib

@st.cache_resource
def load_model(city, horizon):
    mr = project.get_model_registry()
    model_obj = mr.get_model(name=f"aqi_rf_{city}_{horizon}")
    model_dir = model_obj.download()
    model_path = os.path.join(model_dir, f"rf_{city}_{horizon}.pkl")
    model = joblib.load(model_path)
    return model

model_24h = load_model(CITY, "24h")

st.write("Model loaded successfully!")

feature_columns = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm2_5_rolling_24h_mean",
    "aqi_change_rate"
]

latest_row = df.dropna(subset=feature_columns).tail(1)

if len(latest_row) > 0:
    X_latest = latest_row[feature_columns]
    prediction_24h = model_24h.predict(X_latest)[0]

    st.subheader(f"{CITY.capitalize()} — 24 Hour PM2.5 Forecast")
    st.metric(label="Predicted PM2.5 (24h from now)", value=f"{prediction_24h:.1f} µg/m³")
else:
    st.error("Not enough recent data to make a prediction.")