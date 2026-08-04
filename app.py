import streamlit as st
import os
import tempfile
import pandas as pd
import joblib
import requests
import plotly.graph_objects as go
from zoneinfo import ZoneInfo
from dotenv import load_dotenv
import hopsworks
import shap

# Page setup
load_dotenv()
st.set_page_config(page_title="AQI Predictor", page_icon="◐", layout="wide", initial_sidebar_state="collapsed")

CITY_COORDS = {
    "karachi": {"lat": 24.8607, "lon": 67.0011},
    "lahore": {"lat": 31.5497, "lon": 74.3436},
    "islamabad": {"lat": 33.6844, "lon": 73.0479},
}
CITIES = list(CITY_COORDS.keys())
HORIZONS = ["24h", "48h", "72h"]
FEATURE_COLUMNS = [
    "aqi", "co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3",
    "hour", "day_of_week", "month", "is_weekend",
    "aqi_lag_1h", "aqi_lag_3h", "aqi_lag_24h",
    "pm25_lag_1h", "pm25_lag_3h", "pm25_lag_24h",
    "pm2_5_rolling_24h_mean",
    "aqi_change_rate"
]
FEATURE_LABELS = {
    "aqi": "Current AQI", "co": "CO", "no": "NO", "no2": "NO2", "o3": "O3", "so2": "SO2",
    "pm2_5": "Current PM2.5", "pm10": "PM10", "nh3": "NH3",
    "hour": "Hour of day", "day_of_week": "Day of week", "month": "Month", "is_weekend": "Is weekend",
    "aqi_lag_1h": "AQI (1h ago)", "aqi_lag_3h": "AQI (3h ago)", "aqi_lag_24h": "AQI (24h ago)",
    "pm25_lag_1h": "PM2.5 (1h ago)", "pm25_lag_3h": "PM2.5 (3h ago)", "pm25_lag_24h": "PM2.5 (24h ago)",
    "pm2_5_rolling_24h_mean": "24h avg PM2.5", "aqi_change_rate": "AQI change rate",
}

# Pollutant chips: a fixed accent color + icon per pollutant, sensor-panel style
POLLUTANT_META = {
    "PM2.5": {"color": "#E63950", "icon": "◆", "key": "pm2_5", "unit": "µg/m³"},
    "PM10":  {"color": "#FF6B35", "icon": "◆", "key": "pm10",  "unit": "µg/m³"},
    "O₃":    {"color": "#3E5C9A", "icon": "◉", "key": "o3",    "unit": "µg/m³"},
    "NO₂":   {"color": "#9B51E0", "icon": "▲", "key": "no2",   "unit": "µg/m³"},
    "SO₂":   {"color": "#F5A623", "icon": "▲", "key": "so2",   "unit": "µg/m³"},
    "CO":    {"color": "#64748B", "icon": "●", "key": "co",    "unit": "µg/m³"},
}

# Minimalist line-icon set (SVG, tinted via `currentColor`) used across the
# header logo and the Current Conditions card, replacing emoji glyphs.
ICONS = {
    "logo": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M7.2 18h9.8a3.8 3.8 0 0 0 .5-7.57A5.3 5.3 0 0 0 7.4 9.4 3.8 3.8 0 0 0 7.2 18Z"/></svg>',
    "thermo": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M14 14.76V3.5a2 2 0 0 0-4 0v11.26a4 4 0 1 0 4 0Z"/></svg>',
    "droplet": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 2.7s6 6.5 6 10.8a6 6 0 1 1-12 0c0-4.3 6-10.8 6-10.8Z"/></svg>',
    "wind": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 8h9.5a2.5 2.5 0 1 0-2.4-3.2"/><path d="M3 12h13a2.5 2.5 0 1 1-2.4 3.2"/><path d="M3 16h7.5a2 2 0 1 1-1.9 2.6"/></svg>',
    "gauge": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M12 12 15.5 8.5"/><circle cx="12" cy="12" r="8.5"/><path d="M12 5.2v1.3M6.2 8.6l1.1.7M5.2 15h1.3M18.8 15h-1.3M17.8 8.6l-1.1.7"/></svg>',
    "feels": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="8" r="3.2"/><path d="M5.5 20a6.5 6.5 0 0 1 13 0"/></svg>',
    "cpu": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="7" y="7" width="10" height="10" rx="1.5"/><path d="M7 3.3v2.3M12 3.3v2.3M17 3.3v2.3M7 18.4v2.3M12 18.4v2.3M17 18.4v2.3M3.3 7h2.3M3.3 12h2.3M3.3 17h2.3M18.4 7h2.3M18.4 12h2.3M18.4 17h2.3"/></svg>',
    "clock": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8.5"/><path d="M12 7.5V12l3 2"/></svg>',
    "target": '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="8"/><circle cx="12" cy="12" r="4"/><circle cx="12" cy="12" r="0.7" fill="currentColor" stroke="none"/></svg>',
}

temp_dir = tempfile.gettempdir()
PKT = ZoneInfo("Asia/Karachi")
OPENWEATHER_KEY = os.getenv("OPENWEATHER_API_KEY")

# Styling 
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;600;700;800&family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

:root {
--bg-page: #EEF4FA;
--bg-card: #FFFFFF;
--ink-900: #10162B;
--ink-600: #4B5468;
--muted: #96A0B5;
--accent: #3E5C9A;
--accent-soft: #E8EDF9;
--border: #E7EBF3;
--sky: #2F80ED;
}

html, body, [class*="css"] { font-family: 'Inter', -apple-system, sans-serif; color: var(--ink-900); }

@keyframes drift-bg {
  0%   { background-position: 0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%; }
  50%  { background-position: 10% 5%, 90% 10%, 95% 95%, 5% 95%, 0% 0%; }
  100% { background-position: 0% 0%, 100% 0%, 100% 100%, 0% 100%, 0% 0%; }
}

.stApp {
  background:
    radial-gradient(1000px circle at 15% 0%, rgba(62,92,154,0.16) 0%, transparent 55%),
    radial-gradient(900px circle at 85% 15%, rgba(155,81,224,0.14) 0%, transparent 50%),
    radial-gradient(1300px circle at 100% 100%, rgba(47,128,237,0.22) 0%, transparent 55%),
    radial-gradient(800px circle at 0% 100%, rgba(18,183,106,0.10) 0%, transparent 45%),
    linear-gradient(160deg, #E8EFFB 0%, #D9E5F5 35%, #CBDBF0 65%, #BCD0EA 100%);
  background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 100% 100%;
  animation: drift-bg 26s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .stApp { animation: none; }
}

.block-container { padding-top: 0.6rem; padding-bottom: 3rem; max-width: 1440px; }
#MainMenu, footer, header { visibility: hidden; }
section[data-testid="stSidebar"] { display: none; }

.mono { font-family: 'JetBrains Mono', monospace; }

.card, div[data-testid="stVerticalBlockBorderWrapper"] {
background: var(--bg-card); border-radius: 18px;
box-shadow: 0 2px 10px rgba(16,22,43,.05), 0 8px 22px rgba(62,92,154,0.07);
transition: .25s ease;
border: 1px solid var(--border);
}
.card { padding: 1.5rem 1.7rem; height: 100%; }
.card:hover { transform: translateY(-2px); box-shadow: 0 10px 26px rgba(16,22,43,.09), 0 4px 12px rgba(62,92,154,0.10); }
div[data-testid="stVerticalBlockBorderWrapper"] { padding: 0.4rem 0.6rem; }

div[class*="st-key-trend_chart_card"],
div[class*="st-key-forecast_chart_card"],
div[class*="st-key-shap_chart_card"] {
background: var(--bg-card);
border-radius: 18px;
padding: 1.3rem 1.5rem 0.6rem 1.5rem;
box-shadow: 0 2px 10px rgba(16,22,43,.05), 0 8px 22px rgba(62,92,154,0.07);
border: 1px solid var(--border);
transition: .25s ease;
}
div[class*="st-key-trend_chart_card"]:hover,
div[class*="st-key-forecast_chart_card"]:hover,
div[class*="st-key-shap_chart_card"]:hover {
transform: translateY(-2px);
box-shadow: 0 10px 26px rgba(16,22,43,.09), 0 4px 12px rgba(62,92,154,0.10);
}

div[class*="st-key-sticky_header"] {
position: sticky;
top: 0;
z-index: 9999;
background: rgba(255,255,255,0.86);
backdrop-filter: blur(16px) saturate(160%);
-webkit-backdrop-filter: blur(16px) saturate(160%);
border-radius: 0 0 22px 22px;
padding: 1.5rem 1.8rem;
margin: 0 -1rem 1.8rem -1rem;
min-height: 92px;
box-shadow: 0 10px 26px rgba(16,22,43,0.07), 0 1px 0 rgba(16,22,43,0.05);
border-bottom: 1px solid rgba(16,22,43,0.06);
background-image:
radial-gradient(circle, rgba(62,92,154,0.06) 1px, transparent 1px),
rgba(255,255,255,0.86);
background-size: 16px 16px, auto;
}
div[class*="st-key-sticky_header"] div[data-testid="stHorizontalBlock"] { align-items: center; }

.app-title-row { display: flex; align-items: center; gap: 0.75rem; }
.app-title-row .logo-mark { flex: 0 0 46px; width: 46px; height: 46px; display: flex; align-items: center; justify-content: center;
color: var(--sky); background: color-mix(in srgb, var(--sky) 12%, white); border-radius: 13px;
box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--sky) 20%, transparent); }
.app-title-row .logo-mark svg { width: 25px; height: 25px; }
.app-title-text { display: flex; flex-direction: column; justify-content: center; }
.app-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.5rem; font-weight: 700;
color: #1B3A6B; letter-spacing: -0.02em; margin: 0; line-height: 1.25; }
.app-sub { font-size: 0.76rem; color: var(--muted); font-weight: 500; margin-top: 0.15rem; }

.meta-right { text-align: right; font-size: 0.78rem; color: var(--muted); line-height: 1.5; }
.meta-right .loc { color: var(--ink-900); font-weight: 700; font-size: 0.9rem; font-family: 'Space Grotesk', sans-serif; }
.meta-right .upd { font-family: 'JetBrains Mono', monospace; font-size: 0.7rem; }

div[class*="st-key-header_controls"] div[data-testid="stHorizontalBlock"] {
gap: 0.4rem !important;
justify-content: flex-end !important;
}
div[class*="st-key-header_controls"] div[data-testid="column"] {
flex: 0 0 auto !important;
width: auto !important;
min-width: 0 !important;
}

div[class*="st-key-refresh_btn_wrap"] .stButton button {
border-radius: 50%; border: 1px solid var(--border); background: var(--accent-soft);
color: var(--accent); font-weight: 700; font-size: 1.25rem; line-height: 1;
width: 46px; height: 46px; min-height: 46px; padding: 0; margin: 0 auto;
display: flex; align-items: center; justify-content: center;
transition: transform .25s ease, background .2s ease, color .2s ease, box-shadow .2s ease;
}
div[class*="st-key-refresh_btn_wrap"] .stButton button:hover {
background: var(--accent); color: #fff; border-color: var(--accent); transform: rotate(50deg);
box-shadow: 0 6px 16px rgba(62,92,154,0.28);
}
div[class*="st-key-refresh_btn_wrap"] .stButton button:active { transform: rotate(160deg); }
div[class*="st-key-refresh_btn_wrap"].spin .stButton button { animation: spin-refresh 0.7s linear infinite; }
@keyframes spin-refresh { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }

div[class*="st-key-loc_btn_wrap"] button {
border-radius: 50%; border: 1px solid var(--border); background: #fff;
color: var(--ink-600); font-weight: 600; font-size: 0.95rem; line-height: 1;
width: 36px; height: 36px; min-height: 36px; padding: 0; margin: 0 auto;
display: flex; align-items: center; justify-content: center;
transition: all .2s ease;
}
div[class*="st-key-loc_btn_wrap"] button:hover {
background: var(--accent-soft); color: var(--accent); border-color: var(--accent);
}
div[data-testid="stPopoverBody"] { min-width: 230px; }
.popover-label { font-size: 0.68rem; font-weight: 700; text-transform: uppercase;
letter-spacing: 0.06em; color: var(--muted); margin-bottom: 0.55rem; }

div[data-testid="stSegmentedControl"] { gap: 0.35rem !important; }
div[data-testid="stSegmentedControl"] label { font-weight: 600 !important; font-size: 0.84rem !important; }
div[data-testid="stSegmentedControl"] button,
div[data-testid="stSegmentedControl"] [role="radio"] {
border-radius: 999px !important;
border: 1px solid var(--border) !important;
background: #fff !important;
color: var(--ink-600) !important;
transition: all .2s ease !important;
box-shadow: none !important;
}
div[data-testid="stSegmentedControl"] button:hover,
div[data-testid="stSegmentedControl"] [role="radio"]:hover {
border-color: var(--accent) !important;
color: var(--accent) !important;
background: var(--accent-soft) !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"],
div[data-testid="stSegmentedControl"] button[aria-pressed="true"],
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] {
background: var(--accent) !important;
border-color: var(--accent) !important;
color: #fff !important;
box-shadow: 0 4px 12px rgba(62,92,154,0.3) !important;
}
div[data-testid="stSegmentedControl"] button[aria-checked="true"] p,
div[data-testid="stSegmentedControl"] [role="radio"][aria-checked="true"] p { color: #fff !important; }

.card-label { font-family: 'Inter', -apple-system, sans-serif; color: var(--muted); font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
letter-spacing: 0.07em; margin-bottom: 0.6rem; display: flex; align-items: center; gap: 0.4rem; }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; box-shadow: 0 0 0 4px currentColor22; }

.aqi-badge { font-size: 0.72rem; font-weight: 700; padding: 0.28rem 0.75rem; border-radius: 100px; display: inline-block; }
.aqi-delta { font-size: 0.8rem; color: var(--ink-600); margin-top: 0.7rem; font-family: 'JetBrains Mono', monospace; }

.aqi-ring-wrap { display: flex; justify-content: center; padding: 0.4rem 0 0.2rem 0; }
.aqi-ring {
--pct: 0; --rc: #999;
width: 210px; height: 210px; border-radius: 50%;
background: conic-gradient(var(--rc) calc(var(--pct) * 1%), #EEF1F7 0);
display: flex; align-items: center; justify-content: center;
position: relative;
box-shadow: inset 0 0 0 1px rgba(16,22,43,0.03);
}
.aqi-ring::after {
content: ""; position: absolute; inset: 0; border-radius: 50%;
box-shadow: 0 0 26px 2px color-mix(in srgb, var(--rc) 35%, transparent);
opacity: 0.55; pointer-events: none;
}
.aqi-ring-inner {
width: 166px; height: 166px; border-radius: 50%; background: var(--bg-card);
display: flex; flex-direction: column; align-items: center; justify-content: center;
box-shadow: inset 0 2px 10px rgba(16,22,43,.05);
z-index: 1;
}
.aqi-value { font-family: 'Space Grotesk', sans-serif; font-size: 3rem; font-weight: 800; line-height: 1; color: var(--rc); }
.aqi-unit { font-size: 0.66rem; color: var(--muted); letter-spacing: 0.14em; text-transform: uppercase; margin-top: 0.35rem; font-weight: 700; }

.pollutant-card { padding: 1.15rem 0.6rem; text-align: center; position: relative; overflow: hidden; }
.pollutant-card::before { content: ""; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: var(--pc); }
.pollutant-icon { font-size: 0.85rem; color: var(--pc); margin-bottom: 0.3rem; }
.pollutant-value { font-family: 'Space Grotesk', sans-serif; font-size: 1.55rem; font-weight: 800; margin: 0; color: var(--ink-900); }
.pollutant-unit { color: var(--muted); font-size: 0.64rem; margin: 0 0 0.4rem 0; font-family: 'JetBrains Mono', monospace; }
.pollutant-name { font-size: 0.74rem; color: var(--pc); font-weight: 700; text-transform: uppercase; letter-spacing: 0.04em; }

.weather-card { border-top: 3px solid var(--sky);
background: linear-gradient(165deg, color-mix(in srgb, var(--sky) 7%, white) 0%, var(--bg-card) 55%);
min-height: 340px; display: flex; flex-direction: column; justify-content: center; }
.weather-hero { display: flex; align-items: center; gap: 0.4rem; padding-bottom: 1.1rem;
margin-bottom: 1.05rem; border-bottom: 1px solid var(--border); }
.weather-hero img { width: 58px; height: 58px; margin: -6px -4px -6px -10px;
filter: drop-shadow(0 3px 8px rgba(47,128,237,0.22)); }
.weather-hero-main { line-height: 1.15; }
.weather-hero-temp { font-family: 'Space Grotesk', sans-serif; font-size: 2.15rem; font-weight: 800; color: var(--sky); }
.weather-hero-desc { font-size: 0.78rem; color: var(--muted); font-weight: 600; margin-top: 0.15rem; }
.weather-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 1.05rem 0.5rem; }
.weather-tile { display: flex; align-items: center; gap: 0.6rem; }
.weather-tile-icon { flex: 0 0 32px; width: 32px; height: 32px; border-radius: 9px;
background: color-mix(in srgb, var(--sky) 13%, white); color: var(--sky);
display: flex; align-items: center; justify-content: center;
box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--sky) 18%, transparent); }
.weather-tile-icon svg { width: 16px; height: 16px; }
.weather-tile-value { font-family: 'JetBrains Mono', monospace; font-weight: 700; font-size: 0.87rem; color: var(--ink-900); line-height: 1.1; }
.weather-tile-label { font-size: 0.65rem; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; margin-top: 0.12rem; }

.health-card { border-left: 4px solid var(--hc); position: relative; overflow: hidden; }
.health-card::after { content: ""; position: absolute; top: -40px; right: -40px; width: 140px; height: 140px;
    border-radius: 50%; background: color-mix(in srgb, var(--hc) 12%, transparent); pointer-events: none; }
.health-headline { font-family: 'Space Grotesk', sans-serif; font-size: 1.15rem; font-weight: 700;
    color: var(--hc); margin: 0.1rem 0 1rem 0; letter-spacing: -0.01em; }
.health-list { display: flex; flex-direction: column; gap: 0.7rem; position: relative; z-index: 1; }
.health-item { display: flex; align-items: flex-start; gap: 0.85rem; }
.health-icon { flex: 0 0 38px; width: 38px; height: 38px; border-radius: 11px; display: flex;
    align-items: center; justify-content: center; font-size: 1.1rem; font-weight: 700; color: var(--hc);
    background: color-mix(in srgb, var(--hc) 14%, white);
    box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--hc) 20%, transparent); }
.health-text { font-size: 0.87rem; line-height: 1.55; color: var(--ink-600); padding-top: 0.45rem; }

.forecast-card { border-top: 3px solid var(--accent-c); padding-top: 1.3rem; background:
linear-gradient(180deg, color-mix(in srgb, var(--accent-c) 5%, white) 0%, var(--bg-card) 60%); }
.forecast-label { font-size: 0.72rem; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.06em; }
.forecast-date { font-size: 0.76rem; color: var(--muted); margin-top: 0.1rem; font-family: 'JetBrains Mono', monospace; }
.forecast-aqi-row { display: flex; align-items: baseline; gap: 0.55rem; margin: 0.55rem 0 0.5rem 0; }
.forecast-aqi { font-family: 'Space Grotesk', sans-serif; font-size: 2.2rem; font-weight: 800; letter-spacing: -0.02em; }
.trend-tag { font-size: 0.68rem; font-weight: 700; padding: 0.16rem 0.6rem; border-radius: 6px; }
.forecast-sub { font-size: 0.75rem; color: var(--ink-600); margin-top: 0.6rem; font-family: 'JetBrains Mono', monospace; }
.rmse-note { font-size: 0.7rem; color: var(--muted); margin-top: 0.3rem; }

div[class*="st-key-trajectory_row"] div[data-testid="stHorizontalBlock"] { align-items: stretch !important; }
div[class*="st-key-trajectory_row"] div[data-testid="column"] { display: flex !important; height: auto; }
div[class*="st-key-trajectory_row"] div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
  display: flex !important; flex-direction: column !important; flex: 1 !important; height: 100% !important;
}
div[class*="st-key-trajectory_row"] div[class*="st-key-forecast_chart_card"] {
  flex: 1 !important; display: flex !important; flex-direction: column !important; justify-content: center !important;
}
div[class*="st-key-trajectory_row"] div[class*="st-key-forecast_chart_card"] > div[data-testid="stVerticalBlock"] {
  flex: 1 !important; display: flex !important; flex-direction: column !important; justify-content: center !important;
}
div[class*="st-key-trajectory_row"] .sys-card { flex: 1 !important; height: auto !important; min-height: 0 !important; }

.sys-card { border-top: 3px solid var(--accent);
background: linear-gradient(165deg, color-mix(in srgb, var(--accent) 6%, white) 0%, var(--bg-card) 55%);
padding-top: 1.4rem; display: flex; flex-direction: column; justify-content: space-between; }
.sys-row { display: flex; justify-content: space-between; align-items: center; padding: 0.85rem 0;
border-bottom: 1px solid var(--border); font-size: 0.9rem; }
.sys-row:last-child { border-bottom: none; }
.sys-label { display: flex; align-items: center; gap: 0.7rem; color: var(--ink-600); font-weight: 500; }
.sys-icon { flex: 0 0 30px; width: 30px; height: 30px; border-radius: 9px;
background: color-mix(in srgb, var(--accent) 13%, white); color: var(--accent);
display: flex; align-items: center; justify-content: center;
box-shadow: inset 0 0 0 1px color-mix(in srgb, var(--accent) 18%, transparent); }
.sys-icon svg { width: 15px; height: 15px; }
.sys-value { font-weight: 700; color: var(--ink-900); font-family: 'JetBrains Mono', monospace; font-size: 0.92rem; }

.alert-card { display: flex; align-items: center; gap: 0.9rem; border-left: 4px solid var(--ac); }
.alert-icon { font-size: 1.3rem; color: var(--ac); }
.alert-title { font-weight: 800; font-size: 0.95rem; color: var(--ac); font-family: 'Space Grotesk', sans-serif; }
.alert-body { font-size: 0.84rem; color: #000000; font-weight: 500; margin-top: 0.15rem; }

.section-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.2rem; font-weight: 700; color: var(--ink-900);
margin-bottom: 0.1rem; margin-top: 2.5rem; letter-spacing: -0.01em; display:flex; align-items:center; gap:0.5rem; }
.section-title::before { content: ""; width: 8px; height: 8px; border-radius: 2px; background: var(--accent); display:inline-block; }
.section-sub { color: var(--ink-600); font-weight: 600; font-size: 0.82rem; margin-bottom: 1.1rem;
text-shadow: 0 1px 0 rgba(255,255,255,0.5); }
</style>
""", unsafe_allow_html=True)


# AQI helpers
def pm25_to_aqi(pm25):
    breakpoints = [
        (0.0, 12.0, 0, 50), (12.1, 35.4, 51, 100), (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200), (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400), (350.5, 500.4, 401, 500),
    ]
    pm25 = max(0.0, pm25)
    for c_lo, c_hi, i_lo, i_hi in breakpoints:
        if c_lo <= pm25 <= c_hi:
            return round((i_hi - i_lo) / (c_hi - c_lo) * (pm25 - c_lo) + i_lo)
    return 500


def aqi_category(aqi):
    if aqi <= 50: return "Good", "#12B76A"
    elif aqi <= 100: return "Moderate", "#F5A623"
    elif aqi <= 150: return "Unhealthy (Sensitive)", "#FF6B35"
    elif aqi <= 200: return "Unhealthy", "#E63950"
    elif aqi <= 300: return "Very Unhealthy", "#9B51E0"
    else: return "Hazardous", "#6B1E23"


def health_advice(aqi):
    """Returns (headline, [(icon, text), ...]) tailored to the severity tier.
    Icons are a small monochrome glyph set (not emoji) so they tint with the
    severity color instead of clashing with it: ✓ do/safe, △ caution,
    ⌂ shelter/ventilation, ✕ avoid."""
    if aqi <= 50:
        return "Safe to be outside", [
            ("✓", "Air quality is satisfactory — enjoy normal outdoor activities without restriction."),
            ("✓", "No precautions are necessary for the general public."),
            ("⌂", "A good day for outdoor exercise or ventilating indoor spaces."),
        ]
    elif aqi <= 100:
        return "Generally fine, minor caution", [
            ("✓", "Air quality is acceptable for most people."),
            ("△", "Unusually sensitive individuals should consider limiting prolonged or heavy exertion outdoors."),
            ("⌂", "Ventilating your home is generally fine at this level."),
        ]
    elif aqi <= 150:
        return "Sensitive groups take note", [
            ("△", "Children, older adults, and those with asthma or heart/lung conditions should reduce prolonged outdoor exertion."),
            ("✓", "The general public can continue normal activities, but watch for coughing or shortness of breath."),
            ("⌂", "Consider keeping windows closed during peak traffic hours."),
        ]
    elif aqi <= 200:
        return "Limit time outdoors", [
            ("△", "Everyone may begin to experience health effects; sensitive groups may experience more serious effects."),
            ("⌂", "Limit prolonged outdoor exertion — move exercise indoors where possible."),
            ("✓", "Sensitive individuals should wear a well-fitted N95 mask if going outside is unavoidable."),
            ("⌂", "Keep windows closed; use an air purifier indoors if available."),
        ]
    elif aqi <= 300:
        return "Health alert — stay indoors", [
            ("△", "Everyone may experience more serious health effects at this level."),
            ("✕", "Avoid all outdoor physical exertion."),
            ("⌂", "Sensitive groups should remain indoors and minimise activity."),
            ("✓", "Wear a well-fitted N95 mask if you must go outside."),
        ]
    else:
        return "Health emergency", [
            ("△", "The entire population is at increased risk."),
            ("✕", "Avoid all outdoor activity — remain indoors with windows and doors closed."),
            ("✓", "Use an air purifier if available, and seek medical attention if experiencing breathing difficulty."),
            ("△", "Sensitive groups should consider relocating temporarily if conditions persist."),
        ]


def trend_tag(current, predicted):
    diff = predicted - current
    if diff > 5: return "Rising", "#FCE4E7", "#E63950"
    elif diff < -5: return "Improving", "#E4F6ED", "#12B76A"
    else: return "Stable", "#EEF1F7", "#7A8299"


def get_alert_level(current_aqi, predicted_aqis):
    worst_aqi = max([current_aqi] + list(predicted_aqis.values()))
    worst_cat, worst_color = aqi_category(worst_aqi)
    if worst_aqi > 150:
        return worst_aqi, worst_cat, worst_color
    return None, None, None


def render_aqi_ring(aqi, color):
    pct = max(2, min(100, round(aqi / 500 * 100)))
    return (f'<div class="aqi-ring-wrap"><div class="aqi-ring" style="--pct:{pct}; --rc:{color};">'
            f'<div class="aqi-ring-inner"><div class="aqi-value" style="color:{color};">{aqi}</div>'
            f'<div class="aqi-unit">AQI Index</div></div></div></div>')


def styled_plotly_layout(fig, height):
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter", size=12, color="#4B5468"),
        xaxis=dict(showgrid=False, zeroline=False, showline=False),
        yaxis=dict(showgrid=True, gridcolor="#EEF1F7", zeroline=False, showline=False),
        hoverlabel=dict(bgcolor="#10162B", font=dict(color="white", family="JetBrains Mono", size=12)),
        showlegend=False,
    )
    return fig


def hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = tuple(int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


def make_trend_area_chart(df_recent, accent):
    fig = go.Figure(go.Scatter(
        x=df_recent.index, y=df_recent["pm2_5"],
        mode="lines", line=dict(color=accent, width=2.5, shape="spline", smoothing=0.4),
        fill="tozeroy", fillcolor=hex_to_rgba(accent, 0.14),
        hovertemplate="%{x|%a %I:%M %p}<br><b>%{y:.1f} µg/m³</b><extra></extra>",
    ))
    return styled_plotly_layout(fig, 230)


def make_forecast_line_chart(labels, values, colors, accent):
    fig = go.Figure(go.Scatter(
        x=labels, y=values, mode="lines+markers+text",
        line=dict(color=accent, width=2.5, dash="dot"),
        marker=dict(size=13, color=colors, line=dict(width=2, color="white")),
        text=[str(v) for v in values], textposition="top center",
        textfont=dict(family="JetBrains Mono", size=12, color="#10162B"),
        hovertemplate="%{x}<br><b>AQI %{y}</b><extra></extra>",
    ))
    fig.update_yaxes(range=[0, max(values) * 1.35 + 10])
    return styled_plotly_layout(fig, 260)


# Hopsworks connection & cached loaders 
@st.cache_resource(show_spinner=False)
def get_hopsworks_project():
    return hopsworks.login(api_key_value=os.getenv("HOPSWORKS_API_KEY"), cert_folder=temp_dir)


@st.cache_data(ttl=300, show_spinner=False)
def load_features(_project, city):
    fs = _project.get_feature_store()
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_resource(show_spinner=False)
def load_model(_project, city, horizon):
    mr = _project.get_model_registry()
    all_versions = mr.get_models(name=f"aqi_rf_{city}_{horizon}")
    latest = max(all_versions, key=lambda m: m.version)
    model_dir = latest.download()
    model_path = os.path.join(model_dir, f"rf_{city}_{horizon}.pkl")
    return joblib.load(model_path)


@st.cache_data(ttl=300, show_spinner=False)
def get_model_rmse(_project, city, horizon):
    mr = _project.get_model_registry()
    all_versions = mr.get_models(name=f"aqi_rf_{city}_{horizon}")
    latest = max(all_versions, key=lambda m: m.version)
    metrics = latest.training_metrics or {}
    return metrics.get("test_rmse")


@st.cache_data(ttl=300, show_spinner=False)
def load_weather(city):
    coords = CITY_COORDS[city]
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": coords["lat"], "lon": coords["lon"], "appid": OPENWEATHER_KEY, "units": "metric"}
    try:
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        weather_info = (data.get("weather") or [{}])[0]
        return {
            "temp": data["main"]["temp"],
            "feels_like": data["main"].get("feels_like", data["main"]["temp"]),
            "humidity": data["main"]["humidity"],
            "pressure": data["main"]["pressure"],
            "wind_speed": (data.get("wind") or {}).get("speed", 0),
            "description": weather_info.get("description", "—").title(),
            "icon": weather_info.get("icon", "01d"),
        }
    except Exception:
        return None


# Sticky header: title (left) · meta (right) · refresh + location (small) 
if "is_refreshing" not in st.session_state:
    st.session_state.is_refreshing = False
if "selected_city_label" not in st.session_state:
    st.session_state.selected_city_label = None

CITY_LABELS = [c.capitalize() for c in CITIES]
LABEL_TO_CITY = dict(zip(CITY_LABELS, CITIES))

header = st.container(key="sticky_header")
with header:
    hcol_title, hcol_meta, hcol_controls = st.columns([2.6, 3.1, 0.5], vertical_alignment="center")
    with hcol_title:
        st.markdown(f"""
<div class="app-title-row">
<span class="logo-mark">{ICONS['logo']}</span>
<div class="app-title-text">
<div class="app-title">Pearls AQI Predictor</div>
<div class="app-sub">Live air quality monitoring &amp; ML forecasts</div>
</div>
</div>
""", unsafe_allow_html=True)
    with hcol_meta:
        meta_placeholder = st.empty()
        meta_placeholder.markdown(
            '<div class="meta-right"><div class="loc">— , Pakistan</div><div class="upd">Loading…</div></div>',
            unsafe_allow_html=True,
        )
    with hcol_controls:
        controls_wrap = st.container(key="header_controls")
        with controls_wrap:
            ctrl_refresh, ctrl_loc = st.columns([1, 1], gap="small")
            with ctrl_refresh:
                refresh_wrap = st.container(key="refresh_btn_wrap")
                with refresh_wrap:
                    refresh_clicked = st.button("↻", use_container_width=True, help="Refresh data")
            with ctrl_loc:
                loc_wrap = st.container(key="loc_btn_wrap")
                with loc_wrap:
                    with st.popover("⌖", use_container_width=True, help="Change city"):
                        st.markdown('<div class="popover-label">Select city</div>', unsafe_allow_html=True)
                        default_label = st.session_state.selected_city_label or CITY_LABELS[0]
                        picked_label = st.segmented_control(
                            "City", CITY_LABELS, default=default_label,
                            label_visibility="collapsed", key="city_segctrl",
                        )
                        if picked_label:
                            st.session_state.selected_city_label = picked_label

selected_label = st.session_state.selected_city_label or CITY_LABELS[0]
selected_city = LABEL_TO_CITY[selected_label].capitalize()

if st.session_state.is_refreshing:
    st.markdown("""
<style>
div[class*="st-key-refresh_btn_wrap"] .stButton button { animation: spin-refresh 0.7s linear infinite; }
</style>
""", unsafe_allow_html=True)

if refresh_clicked:
    st.session_state.is_refreshing = True
    st.cache_data.clear()
    st.rerun()

city = selected_city.lower()

# Load everything
project = get_hopsworks_project()
try:
    df = load_features(project, city)
except Exception:
    st.error("Couldn't load data from Hopsworks right now. Please try refreshing the page.")
    st.stop()

models = {horizon: load_model(project, city, horizon) for horizon in HORIZONS}
rmse_values = {horizon: get_model_rmse(project, city, horizon) for horizon in HORIZONS}
weather = load_weather(city)

latest_row = df.dropna(subset=FEATURE_COLUMNS).tail(1)
if len(latest_row) == 0:
    st.error("Not enough recent data to make a prediction for this city.")
    st.stop()

X_latest = latest_row[FEATURE_COLUMNS]
current_pm25 = latest_row["pm2_5"].values[0]
current_aqi = pm25_to_aqi(current_pm25)
current_category, current_color = aqi_category(current_aqi)

# ambient page wash — a tint of the current severity color layered over the
# base atmosphere gradient, so the whole page quietly reflects conditions.
# Keeps the same slow drift animation as the base .stApp rule above.
st.markdown(f"""
<style>
.stApp {{
    background:
        radial-gradient(1200px circle at 8% -10%, {hex_to_rgba(current_color, 0.20)} 0%, transparent 50%),
        radial-gradient(1400px circle at 100% 100%, {hex_to_rgba(current_color, 0.26)} 0%, transparent 55%),
        radial-gradient(900px circle at 85% 10%, rgba(155,81,224,0.12) 0%, transparent 50%),
        radial-gradient(800px circle at 0% 100%, rgba(47,128,237,0.10) 0%, transparent 45%),
        linear-gradient(160deg, #E8EFFB 0%, #D9E5F5 35%, #CBDBF0 65%, #BCD0EA 100%) !important;
    background-size: 180% 180%, 180% 180%, 180% 180%, 180% 180%, 100% 100%;
    animation: drift-bg 26s ease-in-out infinite;
}}
</style>
""", unsafe_allow_html=True)

last_updated_utc = pd.to_datetime(latest_row["timestamp"].values[0], utc=True)
last_updated_pkt = last_updated_utc.tz_convert(PKT)

predicted_aqis_raw = {h: models[h].predict(X_latest)[0] for h in HORIZONS}
predicted_aqis = {h: round(max(0, min(500, v))) for h, v in predicted_aqis_raw.items()}
forecast_dates_pkt = {h: last_updated_pkt + pd.Timedelta(hours=int(h.replace("h", ""))) for h in HORIZONS}

# fill in the sticky header's location/time now that data is loaded
meta_placeholder.markdown(f"""
<div class="meta-right">
<div class="loc">📍 {selected_city}, Pakistan</div>
<div class="upd">Updated {last_updated_pkt.strftime('%b %d · %I:%M %p')} PKT</div>
</div>
""", unsafe_allow_html=True)

# Hazardous AQI alert
worst_aqi, worst_cat, worst_color = get_alert_level(current_aqi, predicted_aqis)
if worst_aqi:
    when = "now" if worst_aqi == current_aqi else "within the next 72 hours"
    st.markdown(f"""
<div class="card alert-card" style="--ac:{worst_color}; background:{hex_to_rgba(worst_color, 0.05)}; margin-bottom:1.3rem;">
<span class="alert-icon">⚠</span>
<div>
<div class="alert-title">Hazardous Air Quality Alert</div>
<div class="alert-body">AQI is expected to reach <b>{worst_aqi}</b> ({worst_cat}) {when}. Sensitive groups should take precautions.</div>
</div>
</div>
""", unsafe_allow_html=True)

# Current AQI (ring) + Health Guidance 
col_left, col_right = st.columns([1, 1.6], gap="medium")

with col_left:
    st.markdown(f"""
<div class="card">
<div class="card-label"><span class="status-dot" style="background:{current_color}; color:{current_color};"></span>Current Air Quality</div>
{render_aqi_ring(current_aqi, current_color)}
<div style="text-align:center; margin-top:0.9rem;">
<span class="aqi-badge" style="background:{hex_to_rgba(current_color, 0.12)}; color:{current_color};">{current_category}</span>
</div>
<div class="aqi-delta" style="text-align:center;">PM2.5: {current_pm25:.1f} µg/m³</div>
</div>
""", unsafe_allow_html=True)

with col_right:
    headline, tips = health_advice(current_aqi)
    tips_html = "".join([
        f'<div class="health-item"><div class="health-icon">{icon}</div>'
        f'<div class="health-text">{text}</div></div>'
        for icon, text in tips
    ])
    st.markdown(f"""
<div class="card health-card" style="--hc:{current_color};">
<div class="card-label">Health Guidance</div>
<div class="health-headline">{headline}</div>
<div class="health-list">{tips_html}</div>
</div>
""", unsafe_allow_html=True)

# Pollutants + Weather 
st.markdown('<div class="section-title">Current Pollutants</div>', unsafe_allow_html=True)
st.markdown(f'<div class="section-sub">Live concentrations in {selected_city}</div>', unsafe_allow_html=True)

poll_col, weather_col = st.columns([2.4, 1], gap="medium")

with poll_col:
    cols = st.columns(3, gap="small")
    for i, (name, meta) in enumerate(POLLUTANT_META.items()):
        value = latest_row[meta["key"]].values[0]
        with cols[i % 3]:
            st.markdown(f"""
<div class="card pollutant-card" style="--pc:{meta['color']}; margin-bottom:0.8rem;">
<div class="pollutant-icon">{meta['icon']}</div>
<p class="pollutant-value">{value:.1f}</p>
<p class="pollutant-unit">{meta['unit']}</p>
<p class="pollutant-name">{name}</p>
</div>
""", unsafe_allow_html=True)

with weather_col:
    if weather:
        owm_icon_url = f"https://openweathermap.org/img/wn/{weather['icon']}@2x.png"
        st.markdown(f"""
<div class="card weather-card" style="--sky:{current_color};">
<div class="card-label">Current Conditions</div>
<div class="weather-hero">
<img src="{owm_icon_url}" alt="{weather['description']}" />
<div class="weather-hero-main">
<div class="weather-hero-temp">{weather['temp']:.0f}°C</div>
<div class="weather-hero-desc">{weather['description']}</div>
</div>
</div>
<div class="weather-grid">
<div class="weather-tile"><div class="weather-tile-icon">{ICONS['feels']}</div>
<div><div class="weather-tile-value">{weather['feels_like']:.0f}°C</div><div class="weather-tile-label">Feels like</div></div></div>
<div class="weather-tile"><div class="weather-tile-icon">{ICONS['droplet']}</div>
<div><div class="weather-tile-value">{weather['humidity']}%</div><div class="weather-tile-label">Humidity</div></div></div>
<div class="weather-tile"><div class="weather-tile-icon">{ICONS['wind']}</div>
<div><div class="weather-tile-value">{weather['wind_speed']:.1f} m/s</div><div class="weather-tile-label">Wind</div></div></div>
<div class="weather-tile"><div class="weather-tile-icon">{ICONS['gauge']}</div>
<div><div class="weather-tile-value">{weather['pressure']} hPa</div><div class="weather-tile-label">Pressure</div></div></div>
</div>
</div>
""", unsafe_allow_html=True)
    else:
        st.markdown('<div class="card weather-card"><div class="card-label">Current Conditions</div><p style="color:var(--muted); font-size:0.8rem;">Weather data unavailable</p></div>', unsafe_allow_html=True)

# 24h historical trend 
st.markdown('<div class="section-title">24-Hour PM2.5 Trend</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Air quality changes over the last 24 hours (PKT)</div>', unsafe_allow_html=True)

with st.container(key="trend_chart_card"):
    recent_24h = df.tail(24)[["timestamp", "pm2_5"]].copy()
    recent_24h["timestamp"] = recent_24h["timestamp"].dt.tz_convert(PKT)
    recent_24h = recent_24h.set_index("timestamp")
    st.plotly_chart(make_trend_area_chart(recent_24h, current_color), width="stretch", config={"displayModeBar": False})

# AI Forecast
st.markdown('<div class="section-title">3-Day Forecast</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Predicted AQI compared to current conditions</div>', unsafe_allow_html=True)

cols = st.columns(3, gap="medium")
for horizon, col in zip(HORIZONS, cols):
    aqi_val = predicted_aqis[horizon]
    cat, color = aqi_category(aqi_val)
    tag_label, tag_bg, tag_color = trend_tag(current_aqi, aqi_val)
    date_label = forecast_dates_pkt[horizon].strftime('%a, %b %d · %I %p')
    rmse = rmse_values.get(horizon)
    rmse_text = f"±{rmse:.1f} AQI" if rmse is not None else "n/a"
    with col:
        st.markdown(f"""
<div class="card forecast-card" style="--accent-c:{color};">
<div class="forecast-label">+{horizon}</div>
<div class="forecast-date">{date_label} PKT</div>
<div class="forecast-aqi-row">
<div class="forecast-aqi" style="color:{color};">{aqi_val}</div>
<div class="trend-tag" style="background:{tag_bg}; color:{tag_color};">{tag_label}</div>
</div>
<span class="aqi-badge" style="background:{hex_to_rgba(color, 0.12)}; color:{color};">{cat}</span>
<div class="forecast-sub">Model output: {predicted_aqis_raw[horizon]:.1f} (rounded to {aqi_val})</div>
<div class="rmse-note">Model RMSE: {rmse_text}</div>
</div>
""", unsafe_allow_html=True)

#Trend chart + Prediction system info
trajectory_row = st.container(key="trajectory_row")
with trajectory_row:
    chart_col, sys_col = st.columns([2, 1], gap="medium")

    with chart_col:
        st.markdown('<div class="section-title">Predicted AQI Trajectory</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Full trend from now through the next 72 hours</div>', unsafe_allow_html=True)
        with st.container(key="forecast_chart_card"):
            labels = ["Now", "+24h", "+48h", "+72h"]
            values = [current_aqi, predicted_aqis["24h"], predicted_aqis["48h"], predicted_aqis["72h"]]
            point_colors = [aqi_category(v)[1] for v in values]
            st.plotly_chart(
                make_forecast_line_chart(labels, values, point_colors, "#3E5C9A"),
                width="stretch", config={"displayModeBar": False},
            )

    with sys_col:
        st.markdown('<div class="section-title">Prediction System</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Model details</div>', unsafe_allow_html=True)
        rows = "".join([
            f'<div class="sys-row"><span class="sys-label"><span class="sys-icon">{ICONS["target"]}</span>{h} RMSE</span>'
            f'<span class="sys-value">{f"±{rmse_values[h]:.1f}" if rmse_values[h] is not None else "n/a"}</span></div>'
            for h in HORIZONS
        ])
        st.markdown(f"""
<div class="card sys-card">
<div class="sys-row"><span class="sys-label"><span class="sys-icon">{ICONS['cpu']}</span>Model type</span><span class="sys-value">Random Forest</span></div>
<div class="sys-row"><span class="sys-label"><span class="sys-icon">{ICONS['clock']}</span>Forecast horizons</span><span class="sys-value">24 / 48 / 72h</span></div>
{rows}
</div>
""", unsafe_allow_html=True)

# SHAP: Why this prediction 
st.markdown('<div class="section-title">Why This Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="section-sub">Top factors driving the 24h forecast (SHAP feature importance)</div>', unsafe_allow_html=True)

with st.container(key="shap_chart_card"):
    try:
        explainer = shap.TreeExplainer(models["24h"])
        shap_values = explainer.shap_values(X_latest)
        contributions = pd.DataFrame({
            "feature": [FEATURE_LABELS.get(f, f) for f in FEATURE_COLUMNS],
            "value": shap_values[0],
        })
        contributions["abs_value"] = contributions["value"].abs()
        top_features = contributions.sort_values("abs_value", ascending=False).head(10).sort_values("value")

        fig = go.Figure(go.Bar(
            x=top_features["value"], y=top_features["feature"], orientation="h",
            marker_color=[current_color if v > 0 else "#2F80ED" for v in top_features["value"]],
            marker_line_width=0,
            hovertemplate="%{y}<br><b>%{x:.2f} AQI</b><extra></extra>",
        ))
        fig.update_layout(
            height=380, margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Impact on predicted AQI",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=12, color="#4B5468"),
            yaxis=dict(showgrid=False), xaxis=dict(showgrid=True, gridcolor="#EEF1F7", zeroline=True, zerolinecolor="#D8DEEA"),
        )
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        st.caption(f"{current_category}-colored bars push the prediction higher; blue bars pull it lower.")
    except Exception:
        st.info("SHAP explanation unavailable for this model right now.")

if st.session_state.is_refreshing:
    st.session_state.is_refreshing = False
    st.rerun()