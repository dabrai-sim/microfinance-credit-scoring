import subprocess
import pandas as pd
import json

def load_experiments():
    """
    Load all DVC experiments (including committed and queued ones)
    """
    print("Loading all DVC experiments (including past runs)...")

    # Use DVC CLI to get all experiment data in JSON
    result = subprocess.run(
        ["dvc", "exp", "show", "--json", "--all-commits"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print("Error fetching experiments:\n", result.stderr)
        return pd.DataFrame()

    data = json.loads(result.stdout)
    experiments = []

    for exp_name, exp_info in data.items():
        # Skip workspace-only data if no metrics
        metrics = exp_info.get("metrics", {})
        params = exp_info.get("params", {})

        # Read test metrics (if exist)
        test_metrics = metrics.get("metrics/test_metrics.json", {}).get("data", {})
        train_metrics = metrics.get("metrics/train_metrics.json", {}).get("data", {})

        row = {
            "experiment": exp_name,
            "n_estimators": params.get("params.yaml", {}).get("train", {}).get("n_estimators"),
            "max_depth": params.get("params.yaml", {}).get("train", {}).get("max_depth"),
            "min_samples_split": params.get("params.yaml", {}).get("train", {}).get("min_samples_split"),
            "min_samples_leaf": params.get("params.yaml", {}).get("train", {}).get("min_samples_leaf"),
            "max_features": params.get("params.yaml", {}).get("train", {}).get("max_features"),
            "test_accuracy": test_metrics.get("accuracy", 0),
            "test_precision": test_metrics.get("precision", 0),
            "test_recall": test_metrics.get("recall", 0),
            "test_f1": test_metrics.get("f1_score", 0),
            "test_roc_auc": test_metrics.get("roc_auc", 0),
            "train_roc_auc": train_metrics.get("roc_auc", 0),
        }
        experiments.append(row)

    return pd.DataFrame(experiments)

def main():
    df = load_experiments()

    if df.empty:
        print("No experiments found.")
        return

    df = df.sort_values(by="test_roc_auc", ascending=False)
    print("\nEXPERIMENT COMPARISON - Sorted by Test ROC AUC")
    print(df.to_string(index=False))

    df.to_csv("experiments_comparison.csv", index=False)
    print("\nResults saved to experiments_comparison.csv")

    # Identify best experiment
    best_exp = df.iloc[0]
    print("\nBEST EXPERIMENT")
    print(f"Name: {best_exp['experiment']}")
    print(f"Test ROC AUC: {best_exp['test_roc_auc']:.4f}")
    print(f"Test Accuracy: {best_exp['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best_exp['test_f1']:.4f}\n")

    print("Parameters:")
    print(f"  n_estimators: {best_exp['n_estimators']}")
    print(f"  max_depth: {best_exp['max_depth']}")
    print(f"  min_samples_split: {best_exp['min_samples_split']}")
    print(f"  min_samples_leaf: {best_exp['min_samples_leaf']}")
    print(f"  max_features: {best_exp['max_features']}")

    # Save best experiment ID
    with open("best_experiment.txt", "w") as f:
        f.write(best_exp["experiment"])
    print("\nSaved best experiment ID to best_experiment.txt")

if __name__ == "__main__":
    main()

