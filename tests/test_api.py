"""
Tests for FastAPI application
"""
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch
import numpy as np

from src.api.main import app, CreditApplication


@pytest.fixture
def client():
    """Create test client"""
    return TestClient(app)


@pytest.fixture
def sample_application():
    """Sample credit application"""
    return {
        "amt_income_total": 150000.0,
        "amt_credit": 500000.0,
        "amt_annuity": 25000.0,
        "amt_goods_price": 450000.0,
        "age_years": 35.0,
        "employment_years": 5.0,
        "cnt_children": 1,
        "cnt_fam_members": 3,
        "own_car": True,
        "own_realty": True
    }


def test_root_endpoint(client):
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "version" in data
    assert data["version"] == "1.0.0"


def test_health_check_healthy(client):
    """Test health check when model is loaded"""
    with patch('src.api.main.model', Mock()), \
         patch('src.api.main.scaler', Mock()):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True
        assert data["scaler_loaded"] is True


def test_health_check_unhealthy(client):
    """Test health check when model is not loaded"""
    with patch('src.api.main.model', None), \
         patch('src.api.main.scaler', None):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unhealthy"
        assert data["model_loaded"] is False
        assert data["scaler_loaded"] is False


def test_get_metrics_success(client):
    """Test metrics endpoint when file exists"""
    mock_metrics = {
        "accuracy": 0.85,
        "precision": 0.75,
        "recall": 0.80,
        "f1_score": 0.77,
        "roc_auc": 0.90
    }

    with patch('builtins.open', create=True) as mock_open:
        mock_open.return_value.__enter__.return_value.read.return_value = str(mock_metrics)
        with patch('json.load', return_value=mock_metrics):
            response = client.get("/metrics")
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data


def test_get_metrics_not_found(client):
    """Test metrics endpoint when file doesn't exist"""
    with patch('builtins.open', side_effect=FileNotFoundError):
        response = client.get("/metrics")
        assert response.status_code == 404


def test_predict_success(client, sample_application):
    """Test prediction endpoint with valid data"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.7, 0.3]])

    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[0.5] * 13])
    mock_scaler.feature_names_in_ = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'CREDIT_INCOME_RATIO',
        'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO'
    ]

    with patch('src.api.main.model', mock_model), \
         patch('src.api.main.scaler', mock_scaler):
        response = client.post("/predict", json=sample_application)
        assert response.status_code == 200
        data = response.json()
        assert "default_probability" in data
        assert "risk_category" in data
        assert "recommendation" in data
        assert 0 <= data["default_probability"] <= 1


def test_predict_low_risk(client, sample_application):
    """Test prediction for low risk applicant"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.85, 0.15]])

    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[0.5] * 13])
    mock_scaler.feature_names_in_ = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'CREDIT_INCOME_RATIO',
        'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO'
    ]

    with patch('src.api.main.model', mock_model), \
         patch('src.api.main.scaler', mock_scaler):
        response = client.post("/predict", json=sample_application)
        data = response.json()
        assert data["risk_category"] == "Low"
        assert "APPROVED" in data["recommendation"]


def test_predict_medium_risk(client, sample_application):
    """Test prediction for medium risk applicant"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.55, 0.45]])

    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[0.5] * 13])
    mock_scaler.feature_names_in_ = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'CREDIT_INCOME_RATIO',
        'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO'
    ]

    with patch('src.api.main.model', mock_model), \
         patch('src.api.main.scaler', mock_scaler):
        response = client.post("/predict", json=sample_application)
        data = response.json()
        assert data["risk_category"] == "Medium"
        assert "REVIEW REQUIRED" in data["recommendation"]


def test_predict_high_risk(client, sample_application):
    """Test prediction for high risk applicant"""
    mock_model = Mock()
    mock_model.predict_proba.return_value = np.array([[0.3, 0.7]])

    mock_scaler = Mock()
    mock_scaler.transform.return_value = np.array([[0.5] * 13])
    mock_scaler.feature_names_in_ = [
        'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY',
        'AMT_GOODS_PRICE', 'AGE_YEARS', 'EMPLOYMENT_YEARS',
        'CNT_CHILDREN', 'CNT_FAM_MEMBERS', 'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY', 'CREDIT_INCOME_RATIO',
        'ANNUITY_INCOME_RATIO', 'GOODS_CREDIT_RATIO'
    ]
    with patch('src.api.main.model', mock_model), \
         patch('src.api.main.scaler', mock_scaler):
        response = client.post("/predict", json=sample_application)
        data = response.json()
        assert data["risk_category"] == "High"
        assert "REJECTED" in data["recommendation"]


def test_predict_model_not_loaded(client, sample_application):
    """Test prediction when model is not loaded"""
    with patch('src.api.main.model', None), \
         patch('src.api.main.scaler', None):
        response = client.post("/predict", json=sample_application)
        assert response.status_code == 503


def test_predict_invalid_income(client, sample_application):
    """Test prediction with invalid income"""
    sample_application["amt_income_total"] = -1000
    response = client.post("/predict", json=sample_application)
    assert response.status_code == 422


def test_predict_invalid_age(client, sample_application):
    """Test prediction with invalid age"""
    sample_application["age_years"] = 150
    response = client.post("/predict", json=sample_application)
    assert response.status_code == 422


def test_credit_application_validation():
    """Test CreditApplication model validation"""
    # Valid application
    app = CreditApplication(
        amt_income_total=150000,
        amt_credit=500000,
        amt_annuity=25000,
        amt_goods_price=450000,
        age_years=35
    )
    assert app.amt_income_total == 150000
    assert app.cnt_children == 0
    assert app.own_car is False

    # Invalid income
    with pytest.raises(Exception):
        CreditApplication(
            amt_income_total=-1000,
            amt_credit=500000,
            amt_annuity=25000,
            amt_goods_price=450000,
            age_years=35
        )
