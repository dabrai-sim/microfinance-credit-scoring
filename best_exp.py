import pandas as pd
import subprocess
from sklearn.preprocessing import MinMaxScaler

CSV_FILE = "experiments_results.csv"

# 1. Load CSV
df = pd.read_csv(CSV_FILE)

# 2. Keep only successful experiments
df_valid = df.dropna(subset=["Experiment"])
df_valid = df_valid[df_valid["State"].astype(str).str.lower() == "success"]

# 3. Extract all test metric columns
test_metric_cols = [c for c in df_valid.columns if "test_metrics.json" in c.lower()]
if not test_metric_cols:
    raise ValueError("No test metric columns found!")

# 4. Select key metrics (prioritize F1, ROC-AUC, Recall)
priority_metrics = [
    "metrics/test_metrics.json:f1_score",
    "metrics/test_metrics.json:roc_auc",
    "metrics/test_metrics.json:recall",
    "metrics/test_metrics.json:precision",
    "metrics/test_metrics.json:accuracy"
]
available_metrics = [m for m in priority_metrics if m in df_valid.columns]

if not available_metrics:
    raise ValueError("No priority test metrics found in the CSV!")

# 5. Normalize the metrics (to make them comparable)
scaler = MinMaxScaler()
df_valid_scaled = df_valid.copy()
df_valid_scaled[available_metrics] = scaler.fit_transform(df_valid[available_metrics])

# 6. Compute a composite weighted score
weights = {
    "metrics/test_metrics.json:f1_score": 0.4,
    "metrics/test_metrics.json:roc_auc": 0.3,
    "metrics/test_metrics.json:recall": 0.2,
    "metrics/test_metrics.json:precision": 0.05,
    "metrics/test_metrics.json:accuracy": 0.05,
}

df_valid_scaled["composite_score"] = sum(
    df_valid_scaled[m] * weights.get(m, 0) for m in available_metrics
)

# 7. Pick best experiment
best_row = df_valid_scaled.sort_values("composite_score", ascending=False).iloc[0]
best_exp_id = best_row["Experiment"]

# 8. Print detailed breakdown
print(" Best experiment:", best_exp_id)
print(" Composite score:", round(best_row['composite_score'], 4))
print("\n Individual metrics:")
for m in available_metrics:
    print(f"   {m.split(':')[-1]}: {df_valid.loc[df_valid['Experiment'] == best_exp_id, m].values[0]:.4f}")

