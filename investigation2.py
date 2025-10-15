# === INVESTIGATION B: Seasonal Behavioural Changes ===
# ------------------------------------------------------
# Objective: Examine whether bat–rat interactions change between seasons
# Techniques: Regression + MAE, MSE, RMSE, NRMSE, R²

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === PREPARATION ===
os.makedirs("plots", exist_ok=True)

# Load both datasets
dataset1 = pd.read_csv("dataset1.csv")
dataset2 = pd.read_csv("dataset2.csv")

print("\n=== Investigation B: Seasonal Behavioural Changes ===")

# --- Clean and prepare data ---
dataset1['season'] = dataset1['season'].astype(str).str.strip().str.title()
dataset2['month'] = dataset2['month'].astype(str).str.strip().str.title()

# --- Merge behaviour indicators by season ---
bat_season_summary = dataset1.groupby('season').agg({
    'risk': 'mean',
    'reward': 'mean',
    'bat_landing_to_food': 'mean'
}).reset_index()

rat_season_summary = dataset2.groupby('month').agg({
    'rat_arrival_number': 'mean',
    'bat_landing_number': 'mean',
    'rat_minutes': 'mean',
    'food_availability': 'mean'
}).reset_index()

print("\nBat seasonal averages:\n", bat_season_summary)
print("\nRat monthly averages:\n", rat_season_summary.head())

# === MODEL: Predict Bat Risk-Taking using Season ===
# Encode seasons numerically for regression
season_map = {s: i for i, s in enumerate(bat_season_summary['season'].unique())}
dataset1['season_code'] = dataset1['season'].map(season_map)

X = dataset1[['season_code']]
y = dataset1['risk']

# Fit linear regression
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# === Error Metrics ===
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / (y.max() - y.min())
r2 = r2_score(y, y_pred)

print(f"\nModel Performance Metrics:")
print(f"  MAE  = {mae:.4f}")
print(f"  MSE  = {mse:.4f}")
print(f"  RMSE = {rmse:.4f}")
print(f"  NRMSE = {nrmse:.4f}")
print(f"  R²   = {r2:.4f}")

# === Plot 1: Mean Bat Risk and Reward by Season ===
plt.figure(figsize=(8, 5))
plt.bar(bat_season_summary['season'], bat_season_summary['risk'], label="Mean Risk-Taking", alpha=0.7)
plt.bar(bat_season_summary['season'], bat_season_summary['reward'], label="Mean Reward", alpha=0.7)
plt.title("Mean Bat Risk-Taking and Reward by Season")
plt.xlabel("Season")
plt.ylabel("Proportion")
plt.legend()
plt.tight_layout()
plt.savefig("plots/bat_seasonal_risk_reward.png", dpi=300)
plt.close()

# === Plot 2: Rat vs Bat Activity by Month ===
plt.figure(figsize=(8, 5))
plt.plot(rat_season_summary['month'], rat_season_summary['rat_arrival_number'], marker='o', label="Rat Arrivals")
plt.plot(rat_season_summary['month'], rat_season_summary['bat_landing_number'], marker='o', label="Bat Landings")
plt.title("Rat and Bat Activity Across Months")
plt.xlabel("Month")
plt.ylabel("Activity Count")
plt.legend()
plt.tight_layout()
plt.savefig("plots/rat_bat_activity_by_month.png", dpi=300)
plt.close()

# === Plot 3: Regression Fit for Season vs Risk ===
plt.figure(figsize=(8, 5))
plt.scatter(dataset1['season_code'], y, label="Observed Risk", alpha=0.7)
plt.plot(dataset1['season_code'], y_pred, color="red", label="Predicted Risk")
plt.title("Regression Fit: Season vs Bat Risk-Taking")
plt.xlabel("Season Code")
plt.ylabel("Risk (0=avoidance, 1=risk-taking)")
plt.legend()
plt.tight_layout()
plt.savefig("plots/season_vs_risk_regression.png", dpi=300)
plt.close()

# === Interpretation Output ===
print("\n=== Interpretation ===")
print("• A low MAE/MSE/RMSE indicates the model predictions align well with actual risk behaviour.")
print("• The NRMSE normalises error relative to the 0–1 risk scale.")
print("• R² indicates how much variation in bat risk-taking is explained by seasonal change.")
print("• If R² > 0.3 or more, this supports the hypothesis that behaviour shifts significantly with seasons.")
print("\nAll plots saved in the 'plots/' directory.")
