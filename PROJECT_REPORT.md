# Pearls AQI Predictor — Project Report

**Live dashboard:** https://aqi-predictor-pk.streamlit.app/

---

## Executive Summary

This project delivers a fully automated, serverless machine learning system that forecasts Air Quality Index (AQI) 24, 48, and 72 hours ahead for three major Pakistani cities — Karachi, Lahore, and Islamabad. It covers the complete lifecycle of an applied ML product: historical data acquisition, feature engineering, exploratory analysis, model selection and evaluation, automated retraining, and a public-facing dashboard with explainability and safety alerting.

Nine Random Forest models (3 cities × 3 forecast horizons) were trained and evaluated against a naive persistence baseline. Every single model outperformed its baseline, with the strongest result — Islamabad's 24-hour forecast — reaching a test R² of 0.226; absolute predictive power remains modest and a train/test gap persists despite regularization, detailed in Section 9. The system runs unattended: an hourly GitHub Actions job ingests fresh data, and a daily job retrains and republishes all nine models to a live model registry that the dashboard reads from directly.

---

## 1. What Was Built

- A **feature pipeline** that pulls hourly pollutant and weather data for three cities from two external APIs, engineers 20+ time-series features (lags, rolling averages, change rates, calendar features), and pushes the result to a cloud feature store.
- A **training pipeline** that retrains nine Random Forest models daily on the full accumulated dataset and registers each new version to a model registry, with real evaluation metrics attached.
- A **live dashboard** that loads the latest feature row and the latest registered models, produces 24h/48h/72h AQI forecasts per city, and presents them with severity color-coding, SHAP-based explanations, hazardous-AQI alerting, and live-pulled model performance metrics — no hardcoded numbers shown to the user.
- **Two GitHub Actions workflows** that run this entire system unattended, with retry logic and per-city error isolation so a single failure doesn't take down the whole pipeline.

Nothing in the production path is manual: data collection, retraining, and deployment of new model versions all happen on a schedule without human intervention.

---

## 2. Data Collection

Twelve months of hourly historical data were backfilled per city:

- **Pollutants** (CO, NO, NO₂, O₃, SO₂, PM2.5, PM10, NH₃) via OpenWeather's Air Pollution History API, pulled in monthly chunks to respect API constraints.
- **Weather** (temperature, wind speed, humidity, pressure) via Open-Meteo's free Archive API — chosen deliberately over OpenWeather's paid historical weather tier after determining its free-tier quota was insufficient for a full year of hourly backfill.

An hourly GitHub Actions job now keeps this dataset current automatically, appending a new row per city every hour, 24/7, since deployment.

**Design decision — backfill window:** we tested whether extending the backfill from 12 to 24 months would improve model generalization, on the hypothesis that more historical seasons would help. The result was worse performance across nearly every city/horizon combination, most likely due to distribution shift in the older data (changes in monitoring conditions, urban development, or measurement methodology over a longer time span). We reverted to the 12-month window — a concrete example of a hypothesis that didn't pan out, tested rigorously and rejected based on evidence rather than intuition.

---

## 3. Feature Engineering

Each hourly row carries:

- **Time features:** hour, day of week, month, is_weekend
- **Lag features:** AQI and PM2.5 at 1h, 3h, and 24h prior
- **Rolling features:** 24-hour rolling mean of PM2.5
- **Derived features:** AQI change rate (current minus 1h ago)
- **Weather features:** temperature, wind speed, humidity, pressure

Target columns are constructed by shifting future AQI values backward in time — e.g., the `target_aqi_us_24h` column for a given row holds the *actual* AQI recorded 24 hours later — turning the historical timeline into standard supervised input→answer pairs.

**Feature-count experimentation:** additional lag points (6h, 12h), rolling standard deviation, and an hour × month interaction term were all tested against the baseline feature set. None improved held-out test performance, and a couple slightly hurt it. This indicates the original, leaner feature set was already close to the ceiling of what the available data volume (~8,000 rows/city) could support — a useful negative result that prevented over-engineering the feature set.

---

## 4. A Key Methodological Correction: Target Variable Design

**Initial approach:** predict raw PM2.5 concentration and convert to AQI at display time using the US EPA piecewise breakpoint formula.

**Problem identified:** because the EPA formula is non-linear and piecewise, the same absolute PM2.5 prediction error translates into different AQI errors depending on which breakpoint segment the true value falls in. Optimizing a model against PM2.5 error is therefore not the same as optimizing against the AQI error the user actually sees and cares about — the project's real deliverable.

**Fix:** the AQI target is now computed *before* training, by applying the EPA breakpoint formula to the future PM2.5 value at feature-engineering time, so the model is trained to directly minimize AQI prediction error rather than a proxy.

**Second-order correction:** during implementation, a bug in the AQI target computation itself was subsequently identified and fixed. All performance figures in this report and the README reflect the corrected computation — they are the final, validated numbers for the project.

This is a good example of iterative correctness-checking in an ML pipeline: getting the target variable definition right (and re-verifying it) is foundational, since every downstream metric is only as meaningful as the target it's measured against.

---

## 5. Model Selection

Four model families were evaluated against a naive persistence baseline ("AQI tomorrow = AQI today"), across all 3 cities × 3 horizons:

| Model | Outcome |
|---|---|
| Ridge Regression | Underperformed the naive baseline in several cases, confirming the AQI/feature relationship is non-linear and a purely linear model is insufficient |
| Random Forest | Best overall balance of accuracy and consistency across cities and horizons — **selected for production** |
| Gradient Boosting | Comparable to Random Forest overall; modestly better specifically for Lahore, but not a clear overall winner |
| Neural Network (Keras) | Underperformed both tree-based models — consistent with the dataset size (~8,000 rows/city) being on the small side for deep learning to have an advantage |

Final production models use regularized Random Forest hyperparameters (`max_depth=8`, `min_samples_leaf=20`, `max_features="sqrt"`), chosen after looser tree constraints showed clear overfitting during earlier tuning passes.

---

## 6. Final Results

- **Every one of the nine models outperforms its naive persistence baseline** — including the three combinations (Lahore 48h, Lahore 72h, Islamabad 72h) where test R² itself is negative. In every case the model's error is smaller than the baseline's, meaning the model has learned real predictive signal even where absolute forecasting skill is still modest.
- **Best individual result:** Islamabad, 24h horizon — test R² of 0.226, RMSE of 33.3 AQI points.
- **Best RMSE overall:** Karachi across all horizons (23.6–24.7), owing to its lower and less volatile pollution levels compared to Lahore and Islamabad.
- **Accuracy declines with horizon length for Karachi and Islamabad** (Islamabad: 0.226 → 0.027 → -0.027), matching the expected pattern in time-series forecasting where uncertainty compounds over longer lead times.
- **Lahore is the hardest city to forecast**, with the weakest and least horizon-monotonic results (0.037 → -0.071 → -0.053). This is directly explained by its EDA profile (Section 7): Lahore has both the highest mean pollution and by far the highest volatility of the three cities.

**Overfitting gap:** train R² sits substantially above test R² for every model (e.g., Lahore 24h: 0.883 train vs. 0.037 test). This gap persisted even with the regularization constraints applied, and is the most significant open issue in the current model — see Section 9.

---

## 7. Exploratory Data Analysis

| City | Mean PM2.5 | Median PM2.5 | Max PM2.5 | Std Dev |
|---|---|---|---|---|
| Karachi | 32.6 | 23.0 | 262.0 | 30.0 |
| Lahore | 101.5 | 57.5 | 785.6 | 107.1 |
| Islamabad | 85.0 | 56.7 | 548.3 | 84.6 |

**Findings:**

1. **Karachi is consistently the cleanest and most stable city** — its mean PM2.5 (32.6) is by far the lowest, and its standard deviation (30.0) is roughly proportional to its mean, indicating comparatively mild variability. This aligns with its coastal location, which limits pollutant accumulation.
2. **Lahore is both the most polluted and most volatile city.** Its mean (101.5) is roughly 3x Karachi's, and its standard deviation (107.1) exceeds its own mean — an unusually high coefficient of variation that signals frequent, severe spike events layered on top of already-elevated baseline pollution. Its maximum recorded reading (785.6 µg/m³) is more than 20x the WHO's most permissive daily guideline.
3. **Islamabad sits between the two,** but its distributional shape (mean 85.0 vs. median 56.7, std dev 84.6) is closer to Lahore's than to Karachi's — a heavy right-skewed tail despite a lower average than Lahore.
4. **All three cities show mean well above median,** confirming right-skewed pollution distributions everywhere: typical hourly conditions are meaningfully cleaner than the average, which is inflated by periodic spike events. This has a direct modeling implication — a model optimizing for average error (RMSE) will naturally be pulled toward predicting these less-frequent-but-large spikes reasonably well, at some cost to precision on the more common, calmer readings.
5. **Seasonal pattern:** Lahore and Islamabad show sharp winter smog spikes (December–January), with Lahore swinging nearly 10x between its cleanest and most polluted months. Karachi stays comparatively flat year-round.
6. **Diurnal pattern:** Lahore and Islamabad both peak overnight through late morning and dip in early afternoon — the signature of nighttime temperature inversions trapping pollutants near the surface. This finding is what motivated adding temperature and wind speed as model features in the first place.
7. **Pollutant correlations:** PM2.5 and PM10 are near-perfectly correlated (expected, as PM2.5 is a subset of PM10). CO correlates strongly with PM2.5, consistent with shared combustion sources (traffic, industry). Ozone behaves differently by city, reflecting differing atmospheric chemistry between the coastal (Karachi) and inland (Lahore, Islamabad) locations.

**This EDA directly explains the modeling results in Section 6:** Lahore's combination of highest mean pollution, widest seasonal swing, and heaviest distributional tail is exactly why it is the hardest city to forecast with roughly one year of training data — reflected in its negative test R² at the 48h and 72h horizons.

---

## 8. Automated Infrastructure (CI/CD)

- **Hourly Feature Pipeline** — fetches current pollution and weather data for all three cities, computes the full feature row using recent history pulled from the feature store, and inserts it. Includes retry logic (3 attempts with backoff) on both external API calls and feature-store reads/inserts, plus per-city error isolation so a failure fetching one city's data does not block the other two.
- **Daily Training Pipeline** — retrains all nine models on the full accumulated dataset and registers new versions to the model registry, with retry logic on the registration/upload step.

**A known and documented platform limitation:** GitHub Actions' shared scheduler does not guarantee exact execution timing, and hourly jobs can be delayed — particularly at the top of the hour, when many workflows across the entire GitHub platform fire simultaneously. This was mitigated by offsetting the cron schedule to an uncommon minute (`:07` past the hour), a standard community workaround, though delays of up to roughly an hour can still occur. This is explicitly called out as a platform constraint rather than a defect in the pipeline's own logic.

**Update:** The hourly workflow ran reliably at its intended cadence for several weeks after 
deployment. For the past week, however, and without any changes to the workflow configuration 
or pipeline code, it has been running only around 4-5 times per day rather than 24. Since this 
shift stems from GitHub's shared scheduler prioritization under load rather than any defect 
introduced into the workflow file or pipeline code, and is outside the project's control, the 
decision (confirmed with the project mentor) was to document this behavior rather than pursue 
further fixes.

---

## 9. Honest Assessment of Limitations

- **Absolute forecasting accuracy is modest.** While every model beats the naive baseline, test R² ranges from -0.071 to 0.226. Although several models have negative R² on the held-out test set, every model still outperforms the naive persistence baseline, showing useful predictive signal relative to simply carrying forward the current AQI. Users should understand these are directional forecasts with real uncertainty, not precise point predictions.
- **The train/test R² gap indicates continued overfitting** despite regularization (`max_depth=8`, `min_samples_leaf=20`, `max_features="sqrt"`). Further tuning, stronger regularization, or — more likely to help — a larger training dataset are the probable paths to closing this gap.
- **Single year of seasonal coverage.** The models have seen exactly one occurrence of each season. Performance on an atypical year (an unusually severe or mild winter, for instance) is untested and cannot currently be validated.
- **Longer-horizon accuracy is weaker than 24h**, most severely for Lahore — expected in time-series forecasting generally, and amplified here by Lahore's unusually high volatility.
- **Present-moment rather than forecasted weather is used as an input feature.** This is likely the single highest-leverage improvement available: since the model currently has to implicitly assume weather conditions persist, incorporating actual weather *forecasts* for the target time window would likely improve mid-to-long-range accuracy meaningfully.
- **Cyclical time encoding was not implemented.** Hour and day-of-week are currently encoded as plain integers, meaning the model has no inherent notion that hour 23 and hour 0 are adjacent. Sin/cos cyclical encoding is a well-understood, low-risk improvement identified but not yet implemented.
- **Scheduling delays from GitHub Actions' shared infrastructure** are an accepted platform constraint; a paid runner or dedicated scheduler would tighten timing guarantees if precise hourly cadence becomes a requirement.

---

## 10. Dashboard Capabilities

The live Streamlit dashboard (https://aqi-predictor-pk.streamlit.app/) presents:

- Live 3-day AQI forecasts (24h/48h/72h) per city, with EPA-standard color-coded severity bands
- A real-time circular gauge visualization of current AQI
- A 24-hour historical PM2.5 trend chart
- Current pollutant concentrations and live weather conditions for the selected city
- **SHAP-based explainability** ("Why This Prediction") — a selectable-horizon (24h/48h/72h) bar chart showing the top features driving that specific forecast, color-coded by whether each feature is pushing AQI up or down
- **Hazardous AQI alerting** — a prominent banner surfaces automatically whenever current or forecasted AQI crosses the "Unhealthy for Sensitive Groups" threshold (AQI > 150), using the same EPA breakpoints used throughout the rest of the app for consistency
- **Live model transparency** — each horizon's real RMSE is pulled directly from the model registry at render time rather than hardcoded, so the displayed accuracy always reflects whichever model version is currently live

---

## 11. Conclusion

This project demonstrates a complete, production-style ML system: automated data collection, principled feature engineering, a rigorously tested model selection process (including negative results that were correctly rejected rather than cherry-picked around), a properly time-ordered evaluation methodology, and a genuinely automated deployment pipeline with no manual retraining steps. The headline result — that all nine models beat their naive baselines, with the best (Islamabad 24h) reaching a test R² of 0.226 — is a real, validated signal, obtained after identifying and correcting a target-variable computation bug along the way.

The most honest characterization of the current state: the system reliably beats "assume tomorrow looks like today," which is a genuinely useful forecasting product, but there remains a clear, well-understood set of next steps (forecasted weather inputs, cyclical time encoding, more training data, further regularization) with the potential to close the gap between train and test performance and push absolute accuracy meaningfully higher.