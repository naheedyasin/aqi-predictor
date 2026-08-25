# Pearls AQI Predictor

An end-to-end, serverless machine learning pipeline that forecasts Air Quality Index (AQI) for Karachi, Lahore, and Islamabad, Pakistan — 24, 48, and 72 hours ahead. Built with a fully automated feature pipeline, training pipeline, and live web dashboard.

**Live dashboard:** https://aqi-predictor-pk.streamlit.app/

---

## 1. Project Overview

The system continuously collects hourly weather and pollutant data for three cities, engineers time-series features, stores everything in a cloud feature store, trains and evaluates multiple forecasting models, and serves real-time 3-day AQI predictions through an interactive dashboard — with SHAP-based explainability and hazardous AQI alerts.

## 2. Tech Stack

| Component | Technology |
|---|---|
| Data source | OpenWeather Air Pollution API, Open-Meteo (weather) |
| Feature Store / Model Registry | Hopsworks |
| Modeling | scikit-learn (Ridge, Random Forest, Gradient Boosting), TensorFlow/Keras |
| Explainability | SHAP |
| Automation / CI-CD | GitHub Actions |
| Dashboard | Streamlit, Plotly |
| Language | Python |

## 3. Pipeline Architecture

```
OpenWeather API ──┐
                   ├──> feature_engineering.py ──> Hopsworks Feature Store
Open-Meteo API ────┘                                        │
                                                              ▼
                    hourly_pipeline.py (GitHub Actions, hourly)
                    -> fetches current data, computes features, inserts new row
                                                              │
                                                              ▼
                    train.py (GitHub Actions, daily)
                    -> retrains Random Forest per city/horizon, registers to
                       Hopsworks Model Registry
                                                              │
                                                              ▼
                    app/app.py (Streamlit dashboard)
                    -> loads latest features + models, predicts 24/48/72h AQI,
                       shows SHAP explanations, hazardous AQI alerts
```

## 4. Data Collection

- **Historical backfill:** 12 months of hourly pollutant data (CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH3) per city, via OpenWeather's Air Pollution History API, pulled in monthly chunks.
- **Weather backfill:** 12 months of hourly temperature, wind speed, humidity, and pressure per city, via Open-Meteo's free Archive API (chosen over OpenWeather's paid historical weather tier, which has restrictive free-tier quotas unsuitable for backfilling a full year of hourly data).
- **Live collection:** an hourly GitHub Actions job fetches current conditions from both sources and appends new rows to the Feature Store automatically, 24/7.

**Data quantity experiment:** We tested extending the backfill window from 12 to 24 months, hypothesizing more historical seasons would improve model generalization. Results were worse across nearly every city/horizon combination (see Section 7), likely due to distribution shift in the older data. We reverted to the 12-month window.

## 5. Feature Engineering

For each hourly row, the pipeline computes:

- **Time-based features:** hour, day of week, month, is_weekend
- **Lag features:** AQI and PM2.5 at 1h, 3h, and 24h prior
- **Rolling features:** 24-hour rolling mean of PM2.5
- **Derived features:** AQI change rate (current − 1h ago)
- **Weather features:** temperature, wind speed, humidity, pressure (added after initial EDA revealed strong seasonal/diurnal pollution patterns tied to temperature inversions — see Section 9)

Target columns are built by shifting future values backward (e.g., `target_aqi_us_24h` = the actual AQI 24 hours after each row), creating supervised input→answer training pairs from the historical timeline.

## 6. Target Variable: PM2.5 vs. Direct AQI

**Initial approach:** predict raw PM2.5 concentration, convert to AQI via the US EPA breakpoint formula for display.

**Revised approach:** predict the US EPA AQI value directly, computed at feature-engineering time by applying the breakpoint formula to future PM2.5 values before they become the training target.

**Why we changed this:** predicting a proxy (PM2.5) and converting afterward risks compounding error through the EPA formula's non-linear, piecewise structure — the same absolute PM2.5 error translates to different AQI error depending on which breakpoint segment it falls in. Predicting AQI directly is more methodologically defensible and matches the project's literal goal.

**Correction during implementation:** an error in the initial AQI target computation was subsequently identified and fixed. All results reported in Section 7 reflect the corrected target calculation and are the final, validated numbers for this project.

## 7. Model Experimentation

We evaluated four model types against a naive persistence baseline ("tomorrow's AQI = today's AQI"), across all 3 cities and all 3 forecast horizons:

| Model | Result summary |
|---|---|
| Ridge Regression | Underperformed the naive baseline in several cases — confirmed the relationship is non-linear |
| Random Forest | Best overall balance of performance and consistency — selected as the production model |
| Gradient Boosting | Comparable to Random Forest; slightly better for Lahore specifically, but not a clear overall winner |
| Neural Network (Keras) | Underperformed both tree-based models, consistent with having a relatively small dataset (~8,000 rows/city) for deep learning to shine |

### Model Performance

Nine Random Forest models were trained (3 cities × 3 forecast horizons) to predict AQI on the US EPA scale. After observing overfitting with looser tree constraints, the final models used regularized hyperparameters (`max_depth=8`, `min_samples_leaf=20`, `max_features="sqrt"`) to improve generalization.

Test R² ranged from -0.071 to 0.226 across the nine city/horizon combinations, with the strongest result at Islamabad's 24h horizon (R² = 0.226). Critically, every model outperformed its naive persistence baseline (R² range: -0.783 to -0.150), confirming the models learned genuine predictive signal rather than memorizing noise — a meaningful result given how volatile hour-to-hour pollutant data is.

Accuracy declined at longer horizons (e.g., Islamabad: 0.226 → 0.027 → -0.027 across 24h/48h/72h), consistent with the expectation that forecast difficulty compounds over time. The current feature set uses present-moment weather rather than forecasted weather, which is the most likely lever for improving mid-to-long-range accuracy in future work.

Feature-count experiments: adding extra lag points (6h, 12h), rolling standard deviation, and an hour×month interaction term were all tested and none improved test performance — a couple slightly hurt it. The leaner original feature set was already close to optimal for this amount of data.

Evaluation methodology: models are evaluated on a chronological 80/20 split, never a random one (which would leak future rows into training). Each city's data is time-ordered and the most recent 20% is held out as the test set, mirroring how the model is actually used in production.

## 8. Automated Pipelines (CI/CD)

- **Hourly Feature Pipeline** (`src/hourly_pipeline.py`, GitHub Actions, cron: `7 * * * *`): fetches current pollution + weather data for all 3 cities, computes features using recent history from Hopsworks, and inserts the new row. Includes retry logic (3 attempts with backoff) for both external API calls and Hopsworks reads/inserts, and per-city error isolation so one city's failure doesn't block the others.
- **Daily Training Pipeline** (`src/train.py`, GitHub Actions, cron: `0 2 * * *`): retrains all 9 models (3 cities × 3 horizons) on the full accumulated dataset and registers new versions to the Hopsworks Model Registry, with retry logic on the registration/upload step.

**Known limitation:** GitHub Actions' shared scheduler does not guarantee exact execution timing — hourly runs can be delayed, especially at the top of the hour when many workflows across GitHub fire simultaneously. We mitigated this by offsetting the cron schedule to an uncommon minute (`:07`), a common community workaround, though delays of up to ~1 hour can still occur. This is a documented platform limitation, not a defect in our pipeline logic.

## 9. Exploratory Data Analysis

Full analysis in `scripts/eda.py` and `scripts/eda_output/`. Key findings:

| City | Mean PM2.5 | Median PM2.5 | Max PM2.5 | Std Dev |
|---|---|---|---|---|
| Karachi | 32.6 | 23.0 | 262.0 | 30.0 |
| Lahore | 101.5 | 57.5 | 785.6 | 107.1 |
| Islamabad | 85.0 | 56.7 | 548.3 | 84.6 |

1. **PM2.5 distribution:** Karachi is consistently the least polluted (mean 32.6 µg/m³); Lahore is both the most polluted (mean 101.5 µg/m³) and most volatile, with a long tail reaching ~785.6 µg/m³ and a standard deviation (107.1) larger than its mean — a strong right skew. Islamabad sits in between at 85.0 µg/m³, closer to Lahore than to Karachi, and shows a similarly heavy tail relative to its median.
2. **Mean vs. median gap:** all three cities show mean well above median (Karachi 32.6 vs 23.0, Lahore 101.5 vs 57.5, Islamabad 85.0 vs 56.7), confirming pollution readings are right-skewed everywhere — typical conditions are meaningfully cleaner than the average, which is pulled up by periodic spike events.
3. **Seasonal trend:** Lahore and Islamabad show pronounced winter smog spikes (Dec–Jan), with Lahore swinging nearly 10x between its cleanest and most polluted months. Karachi stays comparatively flat year-round, consistent with its coastal location.
4. **Diurnal (hourly) pattern:** Lahore and Islamabad peak overnight through late morning and dip in early afternoon — a signature of nighttime temperature inversions trapping pollutants near the surface. This finding directly motivated adding temperature and wind speed as model features.
5. **Pollutant correlations:** PM2.5/PM10 are near-perfectly correlated (as expected), CO correlates strongly with PM2.5 (shared combustion sources), and ozone behaves differently by city — reflecting differing atmospheric chemistry between coastal and inland locations.

This EDA directly explains the model performance differences observed in Section 7: Lahore's combination of highest mean pollution, widest seasonal swing, and heaviest distributional tail makes it inherently the hardest city to forecast with ~1 year of training data — reflected in its negative test R² at the 48h and 72h horizons.

## 10. Dashboard Features

- Live 3-day AQI forecast (24h/48h/72h) per city, with color-coded severity
- Real-time gauge visualization of current AQI
- 24-hour historical trend chart
- Current pollutant concentrations and live weather conditions
- SHAP explainability ("Why this prediction") — shows the top features driving each forecast horizon (24h/48h/72h, selectable), color-coded by whether they push AQI up or down
- Hazardous AQI alerts — a prominent banner appears whenever current or forecasted AQI exceeds the "Unhealthy for Sensitive Groups" threshold (AQI > 150), using the same EPA breakpoints as the rest of the app for consistency
- Model performance transparency — displays each horizon's real RMSE (pulled live from the Model Registry, not hardcoded)

## 11. Known Limitations & Future Work

- **Test R² remains modest to negative for several city/horizon combinations** (notably Lahore 48h/72h and Islamabad 72h), despite consistently beating the naive persistence baseline. Absolute forecasting accuracy — not just relative improvement over baseline — is the primary area for future work.
- **Overfitting gap between train and test R²** persists despite regularization; further hyperparameter tuning, additional regularization, or more training data are the likely next levers.
- **Single-year seasonal coverage:** the model has only observed one occurrence of each season. Forecasting reliability for atypical years (unusually severe or mild winters) is untested. Multiple years of clean, consistent historical data would be needed to validate cross-year generalization.
- **Longer-horizon accuracy (48h/72h) is weaker than 24h**, especially for Lahore — an expected result in time-series forecasting (uncertainty compounds with horizon length), further amplified by Lahore's high volatility.
- **Weather features were added late in the project;** their full impact on model performance is still being evaluated at time of writing.
- **Cyclical time encoding** (sin/cos transforms for hour and day-of-week) was identified as a potential improvement — representing hour 23 and hour 0 as adjacent rather than maximally different — and is a natural next step.
- **Forecasted (rather than present-moment) weather** as an input feature is the most likely lever for improving mid-to-long-range accuracy.
- **GitHub Actions scheduling delays** are a platform constraint, not a pipeline defect; a paid runner or dedicated scheduler would provide tighter timing guarantees if needed.

## 12. Repository Structure

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
│   ├── app.py                       # Streamlit dashboard
├── scripts/
│   ├── eda.py                       # Exploratory data analysis + chart generation
│   ├── eda_output/                  # Generated EDA charts
│   ├── hopsworks_test.py            # Hopsworks connectivity smoke test
│   └── verify_hopsworks.py          # Feature store / registry verification
├── .github/workflows/               # GitHub Actions CI/CD definitions
└── requirements.txt                 # Root dependencies
```

## 13. Running Locally

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

**Note on deployment:** the live dashboard on Streamlit Cloud reads these same three values from Streamlit's Secrets manager (Settings → Secrets) rather than a `.env` file, since `.env` is gitignored and never deployed. Both are supported — the app checks `st.secrets` first and falls back to `.env`/environment variables for local development.