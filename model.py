"""
model.py
Preprocess data, train models (LinearRegression baseline + RandomForestRegressor),
evaluate and save the best model and scaler.
"""

import os
import joblib
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

DATA_PATH = "dailyActivity_merged.csv"
MODEL_PATH = "model.pkl"
SCALER_PATH = "scaler.pkl"
META_PATH = "meta.json"

# =======================
# Load dataset
# =======================
df = pd.read_csv(DATA_PATH)
df['ActivityDate'] = pd.to_datetime(df['ActivityDate'])

# Select features
feature_candidates = [
    "TotalSteps","TotalDistance",
    "VeryActiveDistance","ModeratelyActiveDistance","LightActiveDistance","SedentaryActiveDistance",
    "VeryActiveMinutes","FairlyActiveMinutes","LightlyActiveMinutes","SedentaryMinutes"
]

features = [f for f in feature_candidates if f in df.columns]
target = "Calories"

assert target in df.columns, f"Target {target} not found in dataset."

# Drop rows with missing values
df_model = df[features + [target]].dropna()
print("Data for modeling shape:", df_model.shape)

X = df_model[features].copy()
y = df_model[target].copy()

# =======================
# Train-test split
# =======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =======================
# Scaling
# =======================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =======================
# Baseline Model (Linear Regression)
# =======================
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)

lr_mae = mean_absolute_error(y_test, y_pred_lr)
# FIX: manual RMSE (old sklearn version)
lr_rmse = mean_squared_error(y_test, y_pred_lr) ** 0.5
lr_r2 = r2_score(y_test, y_pred_lr)

# =======================
# Random Forest Model
# =======================
rf = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
y_pred_rf = rf.predict(X_test_scaled)

rf_mae = mean_absolute_error(y_test, y_pred_rf)
# FIX: manual RMSE
rf_rmse = mean_squared_error(y_test, y_pred_rf) ** 0.5
rf_r2 = r2_score(y_test, y_pred_rf)

# =======================
# Print performance
# =======================
print("\nModel performance:")
print(f"LinearRegression  - MAE: {lr_mae:.3f}, RMSE: {lr_rmse:.3f}, R2: {lr_r2:.3f}")
print(f"RandomForestReg   - MAE: {rf_mae:.3f}, RMSE: {rf_rmse:.3f}, R2: {rf_r2:.3f}")

# Choose best model
best_model = rf if rf_rmse <= lr_rmse else lr
print("Best model:", "RandomForest" if best_model is rf else "LinearRegression")

# =======================
# Save model + scaler
# =======================
joblib.dump(best_model, MODEL_PATH)
joblib.dump(scaler, SCALER_PATH)

print(f"Saved model to {MODEL_PATH} and scaler to {SCALER_PATH}")

# =======================
# Save meta information
# =======================
meta = {
    "features": features,
    "model_type": "RandomForestRegressor" if best_model is rf else "LinearRegression",
    "metrics": {
        "linear": {"mae": lr_mae, "rmse": lr_rmse, "r2": lr_r2},
        "rf": {"mae": rf_mae, "rmse": rf_rmse, "r2": rf_r2}
    },
    "feature_means": X.mean().to_dict()
}

with open(META_PATH, "w") as f:
    json.dump(meta, f, indent=2)

print("Meta saved to", META_PATH)
