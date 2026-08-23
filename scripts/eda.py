# Exploratory Data Analysis: pulls full historical data from Hopsworks for all
# 3 cities and generates charts + written observations identifying trends,
# as required by the project brief.

import os
import tempfile
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import hopsworks
from zoneinfo import ZoneInfo

load_dotenv()

temp_dir = tempfile.gettempdir()
project = hopsworks.login(
    api_key_value=os.getenv("HOPSWORKS_API_KEY"),
    cert_folder=temp_dir
)
fs = project.get_feature_store()

CITIES = ["karachi", "lahore", "islamabad"]
OUTPUT_DIR = "eda_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

sns.set_style("whitegrid")


def load_city_data(city):
    fg = fs.get_feature_group(name=f"aqi_features_{city}", version=1)
    df = fg.read()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["city"] = city.capitalize()
    return df

print("Loading data for all cities...")
all_data = {city: load_city_data(city) for city in CITIES}
combined = pd.concat(all_data.values(), ignore_index=True)
print(f"Loaded {len(combined)} total rows across {len(CITIES)} cities.")

def plot_pm25_distribution(all_data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    for ax, (city, df) in zip(axes, all_data.items()):
        sns.histplot(df["pm2_5"], bins=40, kde=True, ax=ax, color="#3E5C9A")
        ax.set_title(city.capitalize())
        ax.set_xlabel("PM2.5 (µg/m³)")
        ax.axvline(df["pm2_5"].mean(), color="red", linestyle="--", label=f"Mean: {df['pm2_5'].mean():.1f}")
        ax.legend()
    fig.suptitle("PM2.5 Distribution by City", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/pm25_distribution.png", dpi=150)
    plt.close()
    print("Saved: pm25_distribution.png")


plot_pm25_distribution(all_data)

def plot_monthly_trend(all_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for city, df in all_data.items():
        monthly_avg = df.groupby(df["timestamp"].dt.month)["pm2_5"].mean()
        ax.plot(monthly_avg.index, monthly_avg.values, marker="o", label=city.capitalize(), linewidth=2)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Month")
    ax.set_ylabel("Average PM2.5 (µg/m³)")
    ax.set_title("Monthly PM2.5 Trend by City", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/monthly_trend.png", dpi=150)
    plt.close()
    print("Saved: monthly_trend.png")


plot_monthly_trend(all_data)

PKT = ZoneInfo("Asia/Karachi")

def plot_hourly_pattern(all_data):
    fig, ax = plt.subplots(figsize=(12, 6))
    for city, df in all_data.items():
        df_pkt = df.copy()
        df_pkt["timestamp_pkt"] = df_pkt["timestamp"].dt.tz_convert(PKT)
        hourly_avg = df_pkt.groupby(df_pkt["timestamp_pkt"].dt.hour)["pm2_5"].mean()
        ax.plot(hourly_avg.index, hourly_avg.values, marker="o", label=city.capitalize(), linewidth=2)
    ax.set_xticks(range(0, 24))
    ax.set_xlabel("Hour of Day (PKT)")
    ax.set_ylabel("Average PM2.5 (µg/m³)")
    ax.set_title("Hourly PM2.5 Pattern by City (Pakistan Time)", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/hourly_pattern.png", dpi=150)
    plt.close()
    print("Saved: hourly_pattern.png")


plot_hourly_pattern(all_data)

def plot_correlation_heatmap(all_data):
    pollutant_cols = ["co", "no", "no2", "o3", "so2", "pm2_5", "pm10", "nh3"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (city, df) in zip(axes, all_data.items()):
        corr = df[pollutant_cols].corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar=False, vmin=-1, vmax=1)
        ax.set_title(city.capitalize())
    fig.suptitle("Pollutant Correlation Heatmap by City", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/correlation_heatmap.png", dpi=150)
    plt.close()
    print("Saved: correlation_heatmap.png")


plot_correlation_heatmap(all_data)

def plot_city_comparison(all_data):
    summary = pd.DataFrame({
        city: {
            "Mean PM2.5": df["pm2_5"].mean(),
            "Median PM2.5": df["pm2_5"].median(),
            "Max PM2.5": df["pm2_5"].max(),
            "Std Dev": df["pm2_5"].std(),
        }
        for city, df in all_data.items()
    }).T

    fig, ax = plt.subplots(figsize=(10, 6))
    summary[["Mean PM2.5", "Median PM2.5", "Std Dev"]].plot(kind="bar", ax=ax, color=["#3E5C9A", "#12B76A", "#E63950"])
    ax.set_ylabel("PM2.5 (µg/m³)")
    ax.set_title("City Comparison: PM2.5 Statistics", fontsize=14, fontweight="bold")
    ax.set_xticklabels(summary.index, rotation=0)
    ax.legend(title="")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/city_comparison.png", dpi=150)
    plt.close()
    print("Saved: city_comparison.png")

    return summary


summary_stats = plot_city_comparison(all_data)
print("\n=== Summary Statistics ===")
print(summary_stats.round(1))