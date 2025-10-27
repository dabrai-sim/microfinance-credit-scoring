"""
Tests for model training and evaluation
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
import json
from unittest.mock import patch, Mock, MagicMock
from sklearn.ensemble import RandomForestClassifier

from src.models.train import train_model, load_params
from src.models.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_roc_curve,
    plot_precision_recall_curve
)


@pytest.fixture
def sample_train_data():
    """Create sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['TARGET'] = y

    return df


@pytest.fixture
def sample_test_data():
    """Create sample test data"""
    np.random.seed(123)
    n_samples = 30
    n_features = 10

    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 2, n_samples)

    feature_names = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df['TARGET'] = y

    return df


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


@pytest.fixture
def mock_mlflow():
    """Mock MLflow"""
    with patch('src.models.train.mlflow') as mock_mlflow:
        mock_mlflow.start_run = MagicMock()
        mock_mlflow.log_params = MagicMock()
        mock_mlflow.log_metric = MagicMock()
        mock_mlflow.log_artifact = MagicMock()
        mock_mlflow.sklearn.log_model = MagicMock()
        mock_mlflow.set_experiment = MagicMock()
        yield mock_mlflow


def test_load_params_train():
    """Test loading training parameters"""
    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42},
        'train': {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('builtins.open', create=True):
        with patch('yaml.safe_load', return_value=mock_params):
            params = load_params()
            assert 'train' in params
            assert params['train']['n_estimators'] == 100


def test_train_model(sample_train_data, temp_dir, mock_mlflow):
    """Test model training"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # Check model is trained
    assert isinstance(model, RandomForestClassifier)
    assert model.n_estimators == 10

    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics

    # Check files are created
    assert os.path.exists(model_output)
    assert os.path.exists(metrics_output)

    # Check MLflow calls
    assert mock_mlflow.set_experiment.called
    assert mock_mlflow.log_params.called
    assert mock_mlflow.log_metric.called


def test_train_model_with_none_max_depth(sample_train_data, temp_dir, mock_mlflow):
    """Test training with max_depth=None"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')
    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': None,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    assert model.max_depth is None


def test_evaluate_model(sample_test_data, temp_dir):
    """Test model evaluation"""
    test_path = os.path.join(temp_dir, 'test.csv')
    model_path = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')
    plots_dir = os.path.join(temp_dir, 'plots')

    sample_test_data.to_csv(test_path, index=False)

    # Create a simple model
    X = sample_test_data.drop('TARGET', axis=1)
    y = sample_test_data['TARGET']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    import joblib
    joblib.dump(model, model_path)

    with patch('src.models.evaluate.mlflow'):
        metrics = evaluate_model(test_path, model_path, metrics_output, plots_dir)

    # Check metrics
    assert 'accuracy' in metrics
    assert 'precision' in metrics
    assert 'recall' in metrics
    assert 'f1_score' in metrics
    assert 'roc_auc' in metrics

    # Check outputs
    assert os.path.exists(metrics_output)
    assert os.path.exists(plots_dir)
    assert os.path.exists(os.path.join(plots_dir, 'confusion_matrix.png'))
    assert os.path.exists(os.path.join(plots_dir, 'roc_curve.png'))
    assert os.path.exists(os.path.join(plots_dir, 'precision_recall_curve.png'))
    assert os.path.exists(os.path.join(plots_dir, 'prediction_distribution.png'))


def test_plot_confusion_matrix(sample_test_data, temp_dir):
    """Test confusion matrix plotting"""
    output_path = os.path.join(temp_dir, 'confusion_matrix.png')

    y_true = sample_test_data['TARGET'].values
    y_pred = np.random.randint(0, 2, len(y_true))

    plot_confusion_matrix(y_true, y_pred, output_path)

    assert os.path.exists(output_path)


def test_plot_roc_curve(sample_test_data, temp_dir):
    """Test ROC curve plotting"""
    output_path = os.path.join(temp_dir, 'roc_curve.png')

    y_true = sample_test_data['TARGET'].values
    y_proba = np.random.rand(len(y_true))

    plot_roc_curve(y_true, y_proba, output_path)

    assert os.path.exists(output_path)


def test_plot_precision_recall_curve(sample_test_data, temp_dir):
    """Test precision-recall curve plotting"""
    output_path = os.path.join(temp_dir, 'pr_curve.png')

    y_true = sample_test_data['TARGET'].values
    y_proba = np.random.rand(len(y_true))

    plot_precision_recall_curve(y_true, y_proba, output_path)

    assert os.path.exists(output_path)


def test_model_metrics_in_valid_range(sample_train_data, temp_dir, mock_mlflow):
    """Test that model metrics are in valid ranges"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # All metrics should be between 0 and 1
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1


def test_model_feature_importance(sample_train_data, temp_dir, mock_mlflow):
    """Test that feature importance is computed"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # Check that feature importance exists
    assert hasattr(model, 'feature_importances_')
    assert len(model.feature_importances_) == 10
    assert np.sum(model.feature_importances_) > 0


def test_model_predictions_binary(sample_train_data, temp_dir, mock_mlflow):
    """Test that model produces binary predictions"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # Make predictions
    X = sample_train_data.drop('TARGET', axis=1)
    predictions = model.predict(X)

    # Check predictions are binary
    assert set(predictions).issubset({0, 1})
    # Check probabilities sum to 1
    probabilities = model.predict_proba(X)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_evaluate_model_with_perfect_predictions(temp_dir):
    """Test evaluation with perfect predictions"""
    # Create data where model can perfectly predict
    np.random.seed(42)
    n_samples = 100
    X = np.random.randn(n_samples, 5)
    y = (X[:, 0] > 0).astype(int)  # Perfect linear separator

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['TARGET'] = y

    test_path = os.path.join(temp_dir, 'test.csv')
    model_path = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')
    plots_dir = os.path.join(temp_dir, 'plots')

    df.to_csv(test_path, index=False)

    # Train a simple model
    model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=10)
    model.fit(X, y)

    import joblib
    joblib.dump(model, model_path)

    with patch('src.models.evaluate.mlflow'):
        metrics = evaluate_model(test_path, model_path, metrics_output, plots_dir)

    # With a good separator, metrics should be high
    assert metrics['accuracy'] > 0.8
    assert metrics['roc_auc'] > 0.8


def test_train_model_saves_metrics_file(sample_train_data, temp_dir, mock_mlflow):
    """Test that training saves metrics to JSON file"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # Check metrics file exists and can be loaded
    assert os.path.exists(metrics_output)

    with open(metrics_output, 'r') as f:
        saved_metrics = json.load(f)

    assert saved_metrics == metrics


def test_evaluate_model_saves_metrics_file(sample_test_data, temp_dir):
    """Test that evaluation saves metrics to JSON file"""
    test_path = os.path.join(temp_dir, 'test.csv')
    model_path = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')
    plots_dir = os.path.join(temp_dir, 'plots')

    sample_test_data.to_csv(test_path, index=False)

    # Create and save model
    X = sample_test_data.drop('TARGET', axis=1)
    y = sample_test_data['TARGET']
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X, y)

    import joblib
    joblib.dump(model, model_path)

    with patch('src.models.evaluate.mlflow'):
        metrics = evaluate_model(test_path, model_path, metrics_output, plots_dir)

    # Check metrics file exists and can be loaded
    assert os.path.exists(metrics_output)

    with open(metrics_output, 'r') as f:
        saved_metrics = json.load(f)

    assert saved_metrics == metrics


def test_train_model_with_class_weight_balanced(sample_train_data, temp_dir, mock_mlflow):
    """Test that model uses balanced class weights"""
    train_path = os.path.join(temp_dir, 'train.csv')
    model_output = os.path.join(temp_dir, 'model.pkl')
    metrics_output = os.path.join(temp_dir, 'metrics.json')

    sample_train_data.to_csv(train_path, index=False)

    mock_params = {
        'train': {
            'n_estimators': 10,
            'max_depth': 5,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'random_state': 42
        }
    }

    with patch('src.models.train.load_params', return_value=mock_params):
        model, metrics = train_model(train_path, model_output, metrics_output)

    # Check that class_weight is set to balanced
    assert model.class_weight == 'balanced'
