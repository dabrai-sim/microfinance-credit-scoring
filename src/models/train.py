"""
Train credit scoring model with MLflow tracking
"""
import pandas as pd
import yaml
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import json
from pathlib import Path


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def train_model(train_data_path: str, model_output: str, metrics_output: str):
    """
    Train Random Forest model for credit scoring
    """
    params = load_params()
    train_params = params['train']
    # Handle None values for max_depth
    if train_params.get('max_depth') == 'null' or train_params.get('max_depth') is None:
        train_params['max_depth'] = None
    print("Loading training data...")
    train_data = pd.read_csv(train_data_path)

    X_train = train_data.drop('TARGET', axis=1)
    y_train = train_data['TARGET']

    print(f"Training data shape: {X_train.shape}")
    print(f"Number of features: {X_train.shape[1]}")
    print(f"Default rate: {y_train.mean():.2%}")

    # Set MLflow experiment
    mlflow.set_experiment("credit-scoring-experiment")

    with mlflow.start_run(run_name="random_forest_training"):
        print("\nTraining Random Forest model...")

        # Log parameters
        mlflow.log_params(train_params)
        # Train model
        model = RandomForestClassifier(
            n_estimators=train_params['n_estimators'],
            max_depth=train_params['max_depth'],
            min_samples_split=train_params['min_samples_split'],
            min_samples_leaf=train_params['min_samples_leaf'],
            max_features=train_params['max_features'],
            random_state=train_params['random_state'],
            class_weight='balanced',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        # Make predictions on training data
        y_train_pred = model.predict(X_train)
        y_train_proba = model.predict_proba(X_train)[:, 1]

        # Calculate metrics
        train_metrics = {
            'accuracy': float(accuracy_score(y_train, y_train_pred)),
            'precision': float(precision_score(y_train, y_train_pred)),
            'recall': float(recall_score(y_train, y_train_pred)),
            'f1_score': float(f1_score(y_train, y_train_pred)),
            'roc_auc': float(roc_auc_score(y_train, y_train_proba))
        }
        print("\nTraining Metrics:")
        for metric, value in train_metrics.items():
            print(f"  {metric}: {value:.4f}")
            mlflow.log_metric(f"train_{metric}", value)

        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10))
        # Save feature importance
        feature_importance.to_csv('feature_importance.csv', index=False)
        mlflow.log_artifact('feature_importance.csv')
        # Save model
        Path(model_output).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output)
        print(f"\nModel saved to: {model_output}")
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")

        # Save metrics
        Path(metrics_output).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_output, 'w') as f:
            json.dump(train_metrics, f, indent=4)

        print(f"Metrics saved to: {metrics_output}")
        mlflow.log_artifact(metrics_output)
    return model, train_metrics


if __name__ == "__main__":
    train_model(
        train_data_path="data/processed/train.csv",
        model_output="models/credit_model.pkl",
        metrics_output="metrics/train_metrics.json")
