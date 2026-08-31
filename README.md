# Pearls AQI Predictor

An end-to-end machine learning pipeline for forecasting Air Quality Index (AQI) for **Karachi, Lahore, and Islamabad** at **24, 48, and 72 hours ahead**.


**Live dashboard:** https://aqi-predictor-pk.streamlit.app/

---

## 1. Overview

The project automatically collects pollutant and weather data, engineers time-series features, stores them in Hopsworks, retrains forecasting models, and serves predictions through a Streamlit dashboard.

The system includes:

- Automated hourly data ingestion
- Feature engineering and cloud feature storage
- Daily model retraining
- Hopsworks model registry
- 24h/48h/72h AQI forecasting
- SHAP-based prediction explanations
- AQI alerts and live model metrics


## 2. Tech Stack

| Component | Technology |
|---|---|
| Data source | OpenWeather Air Pollution API, Open-Meteo |
| Feature Store | Hopsworks |
| Modeling | scikit-learn, TensorFlow |
| Explainability | SHAP |
| Automation / CI-CD | GitHub Actions |
| Dashboard | Streamlit, Plotly |
| Language | Python |

## 3. Pipeline

```text
OpenWeather ──┐
              ├──> Feature Engineering ──> Hopsworks
Open-Meteo ───┘                              │
                                            ▼
                              Hourly GitHub Actions
                                            │
                                            ▼
                              Daily Model Training
                                            │
                                            ▼
                              Hopsworks Model Registry
                                            │
                                            ▼
                              Streamlit Dashboard
```

## 4. Data Collection

The project uses approximately **12 months of hourly data** for Karachi, Lahore, and Islamabad.

### Historical Data

- **Pollutants:** CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, and NH₃
- **Weather:** Temperature, wind speed, humidity, and pressure
- **Pollution source:** OpenWeather Air Pollution History API
- **Weather source:** Open-Meteo Archive API

Historical pollution data was collected in monthly chunks, while weather data was obtained through Open-Meteo's archive service.

### Live Data

A GitHub Actions workflow runs hourly to fetch the latest pollution and weather observations and append them to the Hopsworks Feature Store.

The pipeline includes retry logic and per-city error handling so that a temporary API or feature-store failure does not stop data collection for the other cities.

### Data Quantity Experiment

A 24-month historical dataset was also tested to determine whether additional historical seasons would improve model performance. The results were generally worse, so the project retained the 12-month dataset.

---

## 5. Feature Engineering

For each hourly observation, the pipeline generates features including:

- **Time:** hour, day of week, month, weekend indicator
- **Lag:** AQI and PM2.5 values from previous hours
- **Rolling:** 24-hour PM2.5 average
- **Derived:** AQI change rate
- **Weather:** temperature, wind speed, humidity, pressure

Future AQI values are shifted to create the prediction targets for 24h, 48h, and 72h forecasting.

AQI is predicted directly using the **US EPA AQI scale** rather than predicting PM2.5 and converting it after prediction.

---

## 6. Modeling

Four model approaches were evaluated:

| Model | Outcome |
|---|---|
| Ridge Regression | Underperformed in several cases |
| Random Forest | Best overall performance; selected for production |
| Gradient Boosting | Competitive but inconsistent |
| Neural Network | Underperformed the tree-based models |

The final system uses **nine Random Forest models**:

- 3 cities
- 3 forecast horizons

The models use regularization to reduce overfitting and are evaluated using a chronological **80/20 train-test split**.

### Final Result

The strongest result was:

**Islamabad — 24h forecast: R² = 0.226**

All nine models outperformed their naive persistence baselines.

Detailed model comparisons and results are provided in the Project Report.

---

## 7. Automated Pipelines

### Hourly Feature Pipeline

`src/hourly_pipeline.py`

Runs hourly through GitHub Actions and:

1. Fetches current pollution and weather data
2. Retrieves recent data from Hopsworks
3. Generates the latest feature row
4. Inserts the row into the Feature Store

Retry logic and per-city error isolation are used to improve reliability.

> **Note:** Although the workflow is configured to run hourly via cron, GitHub Actions' 
> shared scheduler does not guarantee exact timing. The workflow ran as expected for several 
> weeks after deployment, but for the past week has been running approximately 4-5 times per 
> day rather than 24, without any changes made to the workflow file or pipeline code. This is 
> a known GitHub Actions platform limitation and does not indicate a failure in the pipeline 
> logic. Per mentor guidance, this has been documented rather than treated as a bug.

### Daily Training Pipeline

`src/train.py`

Runs daily and:

1. Loads the accumulated feature data
2. Trains the nine forecasting models
3. Evaluates their performance
4. Registers new model versions in Hopsworks

---

## 8. Exploratory Data Analysis

The project includes an EDA pipeline in `scripts/eda.py`.

Key observations included:

- Karachi had the lowest average PM2.5 levels.
- Lahore had the highest average pollution and greatest variability.
- Islamabad showed pollution levels between Karachi and Lahore.
- Lahore and Islamabad showed stronger seasonal winter pollution patterns.
- Pollution levels generally showed clear hourly and seasonal variation.
- PM2.5 and PM10 showed very strong correlation.

These observations helped guide the feature engineering and modeling process.

For the complete EDA and analysis, see the Project Report.

---

## 9. Dashboard

The Streamlit dashboard provides:

- Current AQI
- 24h, 48h, and 72h forecasts
- AQI severity indicators
- 24-hour PM2.5 trend
- Current pollutant concentrations
- Current weather conditions
- SHAP-based prediction explanations
- AQI health alerts
- Live model performance metrics

---

## 10. Repository Structure

```
├── src/
│   ├── fetch_data.py                # Live single-reading fetch (multi-city)
│   ├── backfill_data.py             # Historical pollutant backfill (OpenWeather)
│   ├── backfill_weather.py          # Historical weather backfill (Open-Meteo)
│   ├── feature_engineering.py       # Builds lag/rolling/time/target features
│   ├── add_aqi_target_columns.py    # Adds direct US EPA AQI target columns
│   ├── add_weather_columns.py       # Merges weather features into the feature set
│   ├── push_to_hopsworks.py         # Pushes engineered features to Feature Store
│   ├── hourly_pipeline.py           # Automated hourly feature pipeline
│   └── train.py                     # Automated daily training + model registry
├── app/
│   └── app.py                      # Streamlit dashboard
├── scripts/
│   ├── eda.py                       # Exploratory data analysis + chart generation
│   ├── eda_output/                  # Generated EDA charts
│   ├── hopsworks_test.py            # Hopsworks connectivity smoke test
│   └── verify_hopsworks.py          # Feature store / registry verification
├── .github/workflows/               # GitHub Actions CI/CD definitions
└── requirements.txt                 # Root dependencies
```

## 11. Running Locally

```bash
pip install -r requirements.txt
```

Create a `.env` file in the repo root with:

```
OPENWEATHER_API_KEY=your_key
HOPSWORKS_API_KEY=your_key
HOPSWORKS_PROJECT=your_project_name
```

```bash
streamlit run app/app.py
```

## 12. Further Details

The **Project Report** contains the complete:

- Methodology
- Model experimentation
- Performance analysis
- Exploratory data analysis (EDA)
- Limitations
- Future improvements
- Technical decisions and findings

---

### Deployment

The live dashboard on Streamlit Cloud reads the same environment variables through **Streamlit Secrets** instead of a `.env` file.

For local development, the application supports `.env` files and environment variables. Streamlit Cloud uses its **Settings → Secrets** configuration because `.env` is gitignored and is not deployed.