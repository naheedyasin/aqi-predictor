import os
import tempfile
from dotenv import load_dotenv
import hopsworks

load_dotenv()
temp_dir = tempfile.gettempdir()

project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)

fs = project.get_feature_store()
fg = fs.get_feature_group(name="aqi_features_karachi", version=1)
df = fg.read()

print(f"Total rows: {len(df)}")
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(df.tail(3))