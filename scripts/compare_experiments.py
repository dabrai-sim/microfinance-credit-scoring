"""Compare all DVC experiments"""
import json
import subprocess
import pandas as pd


def get_experiments():
    """Get all experiments from DVC"""
    result = subprocess.run(['dvc', 'exp', 'show', '--json'], capture_output=True, text=True)
    if result.returncode != 0:
        print("Error running dvc exp show")
        return None
    return json.loads(result.stdout)

def parse_experiments(exp_data):
    """Parse experiment data into DataFrame (compatible with new DVC JSON format)"""
    experiments = []

    # exp_data is now a list in latest DVC versions
    if isinstance(exp_data, list):
        for exp in exp_data:
            exp_name = exp.get("name", "unknown")
            exp_dict = {"experiment": exp_name}

            # Extract parameters
            params = exp.get("params", {}).get("params.yaml", {}).get("data", {}).get("train", {})
            exp_dict.update({
                "n_estimators": params.get("n_estimators", ""),
                "max_depth": params.get("max_depth", ""),
                "min_samples_split": params.get("min_samples_split", ""),
                "min_samples_leaf": params.get("min_samples_leaf", ""),
                "max_features": params.get("max_features", "")
            })

            # Extract metrics
            metrics = exp.get("metrics", {})

            test_metrics = metrics.get("metrics/test_metrics.json", {}).get("data", {})
            train_metrics = metrics.get("metrics/train_metrics.json", {}).get("data", {})

            exp_dict.update({
                "test_accuracy": test_metrics.get("accuracy", 0),
                "test_precision": test_metrics.get("precision", 0),
                "test_recall": test_metrics.get("recall", 0),
                "test_f1": test_metrics.get("f1_score", 0),
                "test_roc_auc": test_metrics.get("roc_auc", 0),
                "train_roc_auc": train_metrics.get("roc_auc", 0)
            })

            experiments.append(exp_dict)

    else:
        # backward compatibility for older DVC JSON structure
        for workspace_name, workspace_data in exp_data.items():
            for exp_name, exp_info in workspace_data.items():
                exp_dict = {"experiment": exp_name}

                params = exp_info.get("params", {}).get("params.yaml", {}).get("train", {})
                exp_dict.update({
                    "n_estimators": params.get("n_estimators", ""),
                    "max_depth": params.get("max_depth", ""),
                    "min_samples_split": params.get("min_samples_split", ""),
                    "min_samples_leaf": params.get("min_samples_leaf", ""),
                    "max_features": params.get("max_features", "")
                })

                metrics = exp_info.get("metrics", {})
                test_metrics = metrics.get("metrics/test_metrics.json", {})
                train_metrics = metrics.get("metrics/train_metrics.json", {})

                exp_dict.update({
                    "test_accuracy": test_metrics.get("accuracy", 0),
                    "test_precision": test_metrics.get("precision", 0),
                    "test_recall": test_metrics.get("recall", 0),
                    "test_f1": test_metrics.get("f1_score", 0),
                    "test_roc_auc": test_metrics.get("roc_auc", 0),
                    "train_roc_auc": train_metrics.get("roc_auc", 0)
                })

                experiments.append(exp_dict)

    return pd.DataFrame(experiments)

def main():
    print("Loading DVC experiments...")
    exp_data = get_experiments()
    
    if exp_data is None:
        return
    
    df = parse_experiments(exp_data)
    
    if df.empty:
        print("No experiments found")
        return
    
    df = df.sort_values('test_roc_auc', ascending=False)
    
    print("EXPERIMENT COMPARISON - Sorted by Test ROC AUC")
    
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', lambda x: f'{x:.4f}')
    
    print(df.to_string(index=False))
    
    print("\n")
    print("TOP 5 EXPERIMENTS")

    print(df.head(5).to_string(index=False))
    
    best = df.iloc[0]
    print("\n")
    print("BEST EXPERIMENT")
    print(f"Name: {best['experiment']}")
    print(f"Test ROC AUC: {best['test_roc_auc']:.4f}")
    print(f"Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"Test F1 Score: {best['test_f1']:.4f}")
    print(f"\nParameters:")
    print(f"  n_estimators: {best['n_estimators']}")
    print(f"  max_depth: {best['max_depth']}")
    print(f"  min_samples_split: {best['min_samples_split']}")
    print(f"  min_samples_leaf: {best['min_samples_leaf']}")
    
    df.to_csv('experiments_comparison.csv', index=False)
    print(f"\n Results saved to experiments_comparison.csv")
    
    return df


if __name__ == "__main__":
    main()
