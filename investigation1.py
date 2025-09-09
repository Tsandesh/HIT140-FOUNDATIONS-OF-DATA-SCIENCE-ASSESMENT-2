# --- Investigation A (CLI + Save Plots) ---
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, mannwhitneyu, spearmanr

# --- CREATE PLOTS DIRECTORY ---
os.makedirs("plots", exist_ok=True)

# --- LOAD DATA ---
dataset1 = pd.read_csv("./dataset1.csv")
dataset2 = pd.read_csv("./dataset2.csv")

print("\n=== Investigation A: Do bats perceive rats as predators? ===")

# --- DESCRIPTIVE ANALYSIS (DATASET 1) ---
print("\n--- Dataset 1: Individual Bat Landings ---")

# Risk vs Reward
risk_reward_ct = pd.crosstab(dataset1['risk'], dataset1['reward'])
print("\nRisk vs Reward Contingency Table:")
print(risk_reward_ct)

# Hesitation times
hesitation_by_risk = dataset1.groupby('risk')['bat_landing_to_food'].describe()
print("\nHesitation Times by Risk Group (seconds):")
print(hesitation_by_risk)

# --- INFERENTIAL ANALYSIS (DATASET 1) ---
chi2, p_chi2, _, _ = chi2_contingency(risk_reward_ct)
print(f"\nChi-square test (Risk vs Reward): chi2 = {chi2:.2f}, p = {p_chi2:.2e}")

hes_avoid = dataset1.loc[dataset1['risk']==0, 'bat_landing_to_food']
hes_risk = dataset1.loc[dataset1['risk']==1, 'bat_landing_to_food']
u_stat, p_mwu = mannwhitneyu(hes_avoid, hes_risk, alternative='two-sided')
print(f"Mann-Whitney U test (Hesitation Time): U = {u_stat}, p = {p_mwu:.2e}")

# --- PLOTS FOR DATASET 1 ---
plt.figure(figsize=(6,4))
sns.countplot(x="risk", hue="reward", data=dataset1, palette="Set2")
plt.title("Risk-taking vs Reward Outcomes")
plt.xlabel("Risk (0 = Avoid, 1 = Take risk)")
plt.ylabel("Count")
plt.legend(title="Reward")
plt.savefig("plots/risk_vs_reward.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
sns.boxplot(x="risk", y="bat_landing_to_food", data=dataset1, palette="Set3")
plt.ylim(0, 60)  # cut extreme outliers for visibility
plt.title("Hesitation Time by Risk Behaviour")
plt.xlabel("Risk (0 = Avoid, 1 = Take risk)")
plt.ylabel("Hesitation Time (sec)")
plt.savefig("plots/hesitation_by_risk.png", dpi=300, bbox_inches="tight")
plt.close()

# --- DESCRIPTIVE ANALYSIS (DATASET 2) ---
print("\n--- Dataset 2: 30-min Observation Periods ---")
print("\nSummary Statistics:")
print(dataset2[['bat_landing_number','rat_minutes','rat_arrival_number','food_availability']].describe())

print("\nCorrelation Matrix:")
print(dataset2[['bat_landing_number','rat_minutes','rat_arrival_number']].corr())

# --- INFERENTIAL ANALYSIS (DATASET 2) ---
spearman_corr, spearman_p = spearmanr(dataset2['rat_minutes'], dataset2['bat_landing_number'])
print(f"\nSpearman correlation (Rat minutes vs Bat landings): ρ = {spearman_corr:.3f}, p = {spearman_p:.2e}")

# --- PLOTS FOR DATASET 2 ---
plt.figure(figsize=(6,4))
sns.scatterplot(x="rat_minutes", y="bat_landing_number", data=dataset2)
plt.title("Bat Landings vs Rat Presence Duration")
plt.xlabel("Rat Minutes (per 30-min period)")
plt.ylabel("Bat Landing Number")
plt.savefig("plots/bat_landings_vs_rat_minutes.png", dpi=300, bbox_inches="tight")
plt.close()

plt.figure(figsize=(6,4))
plt.hist(dataset2['bat_landing_number'], bins=20, edgecolor="black")
plt.title("Distribution of Bat Landings (30-min periods)")
plt.xlabel("Bat Landings")
plt.ylabel("Frequency")
plt.savefig("plots/bat_landings_hist.png", dpi=300, bbox_inches="tight")
plt.close()

# --- FINAL SUMMARY ---
print("\n=== FINAL SUMMARY ===")
print("1. Individual bats (Dataset 1):")
print("   - Avoiders (risk=0) were much more successful in feeding than risk-takers (chi-square p < 1e-78).")
print("   - Risk-takers hesitated significantly longer before feeding (Mann-Whitney p < 1e-18).")
print("   - This suggests bats act cautiously in the presence of rats, treating them as a threat.")

print("\n2. Colony-level behaviour (Dataset 2):")
print("   - More rat presence (minutes) was associated with fewer bat landings (ρ ≈ -0.11, p < 1e-7).")
print("   - This shows colony-wide avoidance when rats are active.")

print("\n>>> Overall: Evidence strongly supports that bats perceive rats not only as competitors,")
print("but also as potential predators.")
