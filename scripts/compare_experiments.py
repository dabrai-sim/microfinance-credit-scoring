import os
import json
import subprocess
import pandas as pd


def get_dvc_experiments():
    """Fetch all DVC experiments as JSON."""
    result = subprocess.run(
        ["dvc", "exp", "show", "--json"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print("Error fetching DVC experiments:", result.stderr)
        return []

    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        print("Could not parse JSON output from DVC.")
        return []


def extract_value(exp, path_list):
    """Safely traverse nested dicts."""
    val = exp
    for key in path_list:
        if isinstance(val, dict) and key in val:
            val = val[key]
        else:
            return None
    return val


def compare_experiments():
    print("Loading DVC experiments...")
    experiments = get_dvc_experiments()
    if not experiments:
        print("No experiments found.")
        return

    rows = []
    for exp in experiments:
        exp_data = exp.get("data", {})
        exp_name = exp.get("name") or exp.get("rev")

        # Params
        train_params = extract_value(
            exp_data, ["params", "params.yaml", "data", "train"]
        )

        # Metrics
        train_metrics = extract_value(
            exp_data, ["metrics", "metrics/train_metrics.json", "data"]
        )
        test_metrics = extract_value(
            exp_data, ["metrics", "metrics/test_metrics.json", "data"]
        )

        row = {
            "experiment": exp_name,
            "n_estimators": train_params.get("n_estimators") if train_params else None,
            "max_depth": train_params.get("max_depth") if train_params else None,
            "min_samples_split": train_params.get("min_samples_split") if train_params else None,
            "min_samples_leaf": train_params.get("min_samples_leaf") if train_params else None,
            "max_features": train_params.get("max_features") if train_params else None,
            "test_accuracy": test_metrics.get("accuracy") if test_metrics else 0,
            "test_precision": test_metrics.get("precision") if test_metrics else 0,
            "test_recall": test_metrics.get("recall") if test_metrics else 0,
            "test_f1": test_metrics.get("f1_score") if test_metrics else 0,
            "test_roc_auc": test_metrics.get("roc_auc") if test_metrics else 0,
            "train_roc_auc": train_metrics.get("roc_auc") if train_metrics else 0,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No metrics found in experiments.")
        return

    df_sorted = df.sort_values(by="test_roc_auc", ascending=False)

    print("\nEXPERIMENT COMPARISON - Sorted by Test ROC AUC")
    print(
        df_sorted[
            [
                "experiment",
                "n_estimators",
                "max_depth",
                "min_samples_split",
                "min_samples_leaf",
                "max_features",
                "test_accuracy",
                "test_precision",
                "test_recall",
                "test_f1",
                "test_roc_auc",
                "train_roc_auc",
            ]
        ].to_string(index=False)
    )

    df_sorted.to_csv("experiments_comparison.csv", index=False)
    print("\nResults saved to experiments_comparison.csv")

    # Best experiment summary
    best_exp = df_sorted.iloc[0]
    print("\nBEST EXPERIMENT")
    print(f"Name: {best_exp['experiment']}")
    print(f"Test ROC AUC: {best_exp['test_roc_auc']:.4f}")
    print(f"Test Accuracy: {best_exp['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best_exp['test_f1']:.4f}")
    print("\nParameters:")
    print(f"  n_estimators: {best_exp['n_estimators']}")
    print(f"  max_depth: {best_exp['max_depth']}")
    print(f"  min_samples_split: {best_exp['min_samples_split']}")
    print(f"  min_samples_leaf: {best_exp['min_samples_leaf']}")
    print(f"  max_features: {best_exp['max_features']}")


if __name__ == "__main__":
    compare_experiments()
