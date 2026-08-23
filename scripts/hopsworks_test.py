# Quick connectivity check — confirms Hopsworks login/API key work, without touching any data.

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

print(f"Connected to project: {project.name}")