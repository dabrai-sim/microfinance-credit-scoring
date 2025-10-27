import json
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Now add sys.path modification AFTER imports
sys.path.append('.')
warnings.filterwarnings('ignore')
matplotlib.use('Agg')


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def plot_confusion_matrix(y_true, y_pred, output_path):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_roc_curve(y_true, y_proba, output_path):
    """Plot ROC curve"""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_precision_recall_curve(y_true, y_proba, output_path):
    """Plot Precision-Recall curve"""
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate_model(test_data_path: str, model_path: str, metrics_output: str, plots_dir: str):
    """
    Evaluate trained model on test data
    """

    print("Loading test data...")
    test_data = pd.read_csv(test_data_path)
    X_test = test_data.drop('TARGET', axis=1)
    y_test = test_data['TARGET']
    print(f"Test data shape: {X_test.shape}")
    print(f"Test default rate: {y_test.mean():.2%}")
    print("\nLoading trained model...")
    model = joblib.load(model_path)
    # Make predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    # Calculate metrics
    test_metrics = {
        'accuracy': float(accuracy_score(y_test, y_pred)),
        'precision': float(precision_score(y_test, y_pred)),
        'recall': float(recall_score(y_test, y_pred)),
        'f1_score': float(f1_score(y_test, y_pred)),
        'roc_auc': float(roc_auc_score(y_test, y_proba))
    }
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric}: {value:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Default', 'Default']))

    # Create plots directory
    Path(plots_dir).mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating evaluation plots...")
    plot_confusion_matrix(y_test, y_pred, f"{plots_dir}/confusion_matrix.png")
    plot_roc_curve(y_test, y_proba, f"{plots_dir}/roc_curve.png")
    plot_precision_recall_curve(y_test, y_proba, f"{plots_dir}/precision_recall_curve.png")
    # Prediction distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_proba[y_test == 0], bins=50, alpha=0.5, label='No Default', color='green')
    plt.hist(y_proba[y_test == 1], bins=50, alpha=0.5, label='Default', color='red')
    plt.xlabel('Predicted Probability')
    plt.ylabel('Frequency')
    plt.title('Distribution of Predicted Probabilities')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{plots_dir}/prediction_distribution.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plots saved to: {plots_dir}/")
    # Save metrics
    Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_output, 'w') as f:
        json.dump(test_metrics, f, indent=4)

    print(f"Metrics saved to: {metrics_output}")
    # Log to MLflow
    with mlflow.start_run(run_name="model_evaluation"):
        for metric, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric}", value)
        mlflow.log_artifacts(plots_dir)
        mlflow.log_artifact(metrics_output)
    return test_metrics


if __name__ == "__main__":
    evaluate_model(
        test_data_path="data/processed/test.csv",
        model_path="models/credit_model.pkl",
        metrics_output="metrics/test_metrics.json",
        plots_dir="plots"
    )
