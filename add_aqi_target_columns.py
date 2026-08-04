import os
import tempfile
from dotenv import load_dotenv
import hopsworks
from hsfs.feature import Feature

load_dotenv()

temp_dir = tempfile.gettempdir()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)
fs = project.get_feature_store()

CITIES = ["karachi", "lahore", "islamabad"]

new_features = [
    Feature("target_aqi_us_24h", "double"),
    Feature("target_aqi_us_48h", "double"),
    Feature("target_aqi_us_72h", "double"),
]

for city in CITIES:
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    fg.append_features(new_features)
    print(f"Added new AQI target columns to aqi_features_{city}")