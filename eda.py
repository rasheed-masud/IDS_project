"""
eda.py
Exploratory Data Analysis for dailyActivity_merged.csv
Produces plots and prints summary statistics.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ========== Config ==========
DATA_PATH = "dailyActivity_merged.csv"
OUT_DIR = "plots"
os.makedirs(OUT_DIR, exist_ok=True)

# ========== Load ==========
df = pd.read_csv(DATA_PATH)
# parse date
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

# ========== Basic Info ==========
print("Shape:", df.shape)
print(df.dtypes)
print("\nMissing values:\n", df.isnull().sum())
print("\nSummary statistics:\n", df.describe(include='all'))

# ========== Clean / Quick checks ==========
# Drop duplicates if any
dup_count = df.duplicated().sum()
if dup_count:
    print(f"\nFound {dup_count} duplicate rows. Dropping duplicates.")
    df = df.drop_duplicates()
else:
    print("\nNo duplicate rows found.")

# ========= Feature list ==========
features = [
    "TotalSteps", "TotalDistance",
    "VeryActiveDistance", "ModeratelyActiveDistance", "LightActiveDistance", "SedentaryActiveDistance",
    "VeryActiveMinutes", "FairlyActiveMinutes", "LightlyActiveMinutes", "SedentaryMinutes",
    "Calories"
]

# If any feature missing, reduce list
features = [f for f in features if f in df.columns]
print("\nUsing features:", features)

# ========== 1. Histograms ==========
for col in ["TotalSteps", "TotalDistance", "Calories"]:
    if col in df.columns:
        plt.figure(figsize=(8,5))
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f"Histogram of {col}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"hist_{col}.png"))
        plt.close()

# ========== 2. Boxplots (outliers) ==========
plt.figure(figsize=(10,6))
sns.boxplot(data=df[["TotalSteps","Calories"]].dropna())
plt.title("Boxplots: TotalSteps and Calories")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "box_steps_calories.png"))
plt.close()

# ========== 3. Correlation heatmap ==========
corr = df[features].corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="viridis")
plt.title("Correlation matrix")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "correlation_heatmap.png"))
plt.close()

# ========== 4. Scatter plots ==========
pairs = [("TotalSteps","Calories"), ("TotalDistance","Calories"), ("VeryActiveMinutes","Calories")]
for x,y in pairs:
    if x in df.columns and y in df.columns:
        plt.figure(figsize=(8,6))
        sns.scatterplot(x=df[x], y=df[y])
        plt.title(f"{x} vs {y}")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"scatter_{x}_vs_{y}.png"))
        plt.close()

# ========== 5. Pairplot (subset) ==========
pp_features = [f for f in ["TotalSteps","TotalDistance","VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes","Calories"] if f in df.columns]
if len(pp_features) <= 8:
    sns.pairplot(df[pp_features].dropna().sample(min(500, len(df))))
    plt.savefig(os.path.join(OUT_DIR, "pairplot_subset.png"))
    plt.close()

# ========== 6. Outlier detection using IQR ==========
outlier_report = {}
for col in pp_features:
    series = df[col].dropna()
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = series[(series < lower) | (series > upper)]
    outlier_report[col] = len(outliers)
print("\nOutlier counts by column (IQR method):")
for k,v in outlier_report.items():
    print(f"  {k}: {v}")

# ========== 7. Top N days with highest calories ==========
top_cal = df.sort_values("Calories", ascending=False).head(10)[["ActivityDate","Calories","TotalSteps","TotalDistance"]]
print("\nTop 10 days by Calories:\n", top_cal.to_string(index=False))

# ========== 8. Trend analysis (time series of Calories) ==========
if "ActivityDate" in df.columns:
    df_ts = df.groupby("ActivityDate")["Calories"].mean().reset_index()
    plt.figure(figsize=(12,5))
    sns.lineplot(data=df_ts, x="ActivityDate", y="Calories", marker="o")
    plt.title("Average Calories by Date")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "ts_calories_by_date.png"))
    plt.close()

# ========== Save a cleaned sample for quick inspection ==========
df.sample(10).to_csv("sample_rows.csv", index=False)
print("\nEDA complete. Plots saved in", OUT_DIR)
