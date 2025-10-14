# --- Objective 2: Predictive Modelling (Seasonal Effect) ---

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- LOAD DATA ---
dataset2 = pd.read_csv("dataset2.csv")

print("\n=== Objective 2: Predicting Bat Landing Behaviour Across Seasons ===")

# --- PREPROCESSING ---
dataset2 = dataset2.dropna()

# Encode season/month numerically (if present)
if "season" in dataset2.columns:
    dataset2["season"] = dataset2["season"].astype("category").cat.codes
if "month" in dataset2.columns:
    dataset2["month"] = dataset2["month"].astype("category").cat.codes

# --- DEFINE FEATURES AND TARGET ---
X = dataset2[["rat_minutes", "rat_arrival_number", "food_availability", "hours_after_sunset", "month"]]
y = dataset2["bat_landing_number"]

# --- SPLIT DATA ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- MODEL TRAINING ---
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- METRICS ---
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
nrmse = rmse / (y_test.max() - y_test.min())
r2 = r2_score(y_test, y_pred)

# --- PRINT METRICS ---
print("\n--- Regression Performance Metrics ---")
print(f"Mean Absolute Error (MAE): {mae:.3f}")
print(f"Mean Squared Error (MSE): {mse:.3f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.3f}")
print(f"Normalised RMSE (NRMSE): {nrmse:.3f}")
print(f"Coefficient of Determination (R²): {r2:.3f}")

# --- CREATE DIRECTORY FOR PLOTS ---
os.makedirs("plots", exist_ok=True)

# --- PLOT 1: Predicted vs Actual ---
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("Actual Bat Landings")
plt.ylabel("Predicted Bat Landings")
plt.title("Predicted vs Actual Bat Landing Numbers")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--")
plt.tight_layout()
plt.savefig("plots/predicted_vs_actual.png")
plt.close()

# --- PLOT 2: Residual Plot ---
residuals = y_test - y_pred
plt.figure(figsize=(6,4))
plt.scatter(y_pred, residuals, alpha=0.6)
plt.axhline(0, color="red", linestyle="--")
plt.xlabel("Predicted Bat Landings")
plt.ylabel("Residuals (Actual - Predicted)")
plt.title("Residual Plot")
plt.tight_layout()
plt.savefig("plots/residual_plot.png")
plt.close()

print("\n--- Plots saved in the 'plots/' directory ---")

# --- INTERPRETATION ---
print("\n--- Interpretation Guide ---")
print("• MAE: Average error between predicted and actual values (lower is better).")
print("• MSE: Penalises larger errors more strongly (lower is better).")
print("• RMSE: Same as MSE but in original units (helps intuitive comparison).")
print("• NRMSE: RMSE normalised by data range (0 → perfect, <0.1 → very good).")
print("• R²: Explains variance captured by model (1 = perfect fit, 0 = no fit).")
print("\nExample interpretation:")
print("If R² ≈ 0.7 → model explains ~70% of the variation in bat landings.")
print("If MAE is small → predictions are close on average to real counts.")
print("Residual plot should show points scattered randomly around 0;")
print("patterns would indicate model bias or missing variables.")

print("\n>>> Objective 2 complete: evaluated predictive strength of rat activity and season on bat behaviour.")
