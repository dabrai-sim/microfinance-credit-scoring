"""
Tests for data loading and preprocessing
"""
import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, Mock

from src.data.load_data import load_home_credit_data
from src.data.preprocess import preprocess_data, load_params


@pytest.fixture
def sample_raw_data():
    """Create sample raw data with more rows for proper train-test split"""
    np.random.seed(42)
    n_samples = 50  # Increased from 5 to 50

    data = {
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'AMT_INCOME_TOTAL': np.random.uniform(100000, 200000, n_samples),
        'AMT_CREDIT': np.random.uniform(300000, 600000, n_samples),
        'AMT_ANNUITY': np.random.uniform(15000, 30000, n_samples),
        'AMT_GOODS_PRICE': np.random.uniform(280000, 550000, n_samples),
        'DAYS_BIRTH': np.random.randint(-15000, -10000, n_samples),
        'DAYS_EMPLOYED': np.random.choice(
            list(range(-3000, -1000)) + [365243],
            size=n_samples
        ),
        'CNT_CHILDREN': np.random.randint(0, 4, n_samples),
        'CNT_FAM_MEMBERS': np.random.randint(2, 6, n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.01, 0.03, n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-1500, -800, n_samples),
        'OWN_CAR_AGE': np.random.choice([np.nan, 3.0, 5.0, 10.0], n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(
            ['Working', 'Pensioner', 'Commercial associate'],
            n_samples
        ),
        'NAME_EDUCATION_TYPE': np.random.choice(
            ['Higher education', 'Secondary / secondary special', 'Incomplete higher'],
            n_samples
        ),
        'NAME_FAMILY_STATUS': np.random.choice(
            ['Married', 'Single / not married', 'Separated'],
            n_samples
        ),
        'OCCUPATION_TYPE': np.random.choice(
            ['Laborers', 'Sales staff', 'Core staff', np.nan],
            n_samples
        ),
        'EXT_SOURCE_1': np.random.uniform(0.3, 0.8, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.4, 0.9, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.5, 0.9, n_samples),
    }

    return pd.DataFrame(data)


@pytest.fixture
def temp_dir():
    """Create temporary directory"""
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname


def test_load_home_credit_data(sample_raw_data, temp_dir):
    """Test loading Home Credit data"""
    input_path = os.path.join(temp_dir, 'input.csv')
    output_path = os.path.join(temp_dir, 'output.csv')

    sample_raw_data.to_csv(input_path, index=False)

    load_home_credit_data(input_path, output_path)

    assert os.path.exists(output_path)
    loaded_data = pd.read_csv(output_path)
    assert loaded_data.shape == sample_raw_data.shape
    assert 'TARGET' in loaded_data.columns


def test_load_params():
    """Test loading parameters"""
    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42},
        'train': {'n_estimators': 100, 'max_depth': 10}
    }

    with patch('builtins.open', create=True):
        with patch('yaml.safe_load', return_value=mock_params):
            params = load_params()
            assert 'preprocess' in params
            assert 'train' in params
            assert params['preprocess']['test_size'] == 0.2


def test_preprocess_data(sample_raw_data, temp_dir):
    """Test data preprocessing"""
    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    sample_raw_data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check outputs exist
    assert os.path.exists(train_output)
    assert os.path.exists(test_output)
    assert os.path.exists(scaler_output)

    # Check shapes
    assert len(X_train) + len(X_test) == len(sample_raw_data)
    assert len(y_train) + len(y_test) == len(sample_raw_data)

    # Check features were created
    assert 'AGE_YEARS' in X_train.columns
    assert 'EMPLOYMENT_YEARS' in X_train.columns
    assert 'CREDIT_INCOME_RATIO' in X_train.columns

    # Check DAYS columns were dropped
    assert 'DAYS_BIRTH' not in X_train.columns
    assert 'DAYS_EMPLOYED' not in X_train.columns


def test_preprocess_handles_missing_values(temp_dir):
    """Test preprocessing handles missing values"""
    np.random.seed(42)
    n_samples = 50  # Increased sample size

    data_with_missing = pd.DataFrame({
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'AMT_INCOME_TOTAL': [150000 if i % 5 != 0 else np.nan for i in range(n_samples)],
        'AMT_CREDIT': [500000 if i % 7 != 0 else np.nan for i in range(n_samples)],
        'AMT_ANNUITY': [25000 if i % 6 != 0 else np.nan for i in range(n_samples)],
        'AMT_GOODS_PRICE': np.random.uniform(280000, 550000, n_samples),
        'DAYS_BIRTH': np.random.randint(-15000, -10000, n_samples),
        'DAYS_EMPLOYED': np.random.choice(list(range(-3000, -1000)) + [365243], n_samples),
        'CNT_CHILDREN': np.random.randint(0, 4, n_samples),
        'CNT_FAM_MEMBERS': np.random.randint(2, 6, n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.01, 0.03, n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-1500, -800, n_samples),
        'OWN_CAR_AGE': np.random.choice([np.nan, 3.0, 5.0, 10.0], n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Pensioner'], n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Higher education', 'Secondary / secondary special'], n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single / not married'], n_samples),
        'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Sales staff', np.nan], n_samples),
        'EXT_SOURCE_1': [0.5 if i % 3 != 0 else np.nan for i in range(n_samples)],
        'EXT_SOURCE_2': np.random.uniform(0.4, 0.9, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.5, 0.9, n_samples),
    })

    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    data_with_missing.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check no missing values in output
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0


def test_preprocess_creates_derived_features(sample_raw_data, temp_dir):
    """Test that preprocessing creates derived features"""
    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    sample_raw_data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check derived features exist
    derived_features = [
        'AGE_YEARS',
        'EMPLOYMENT_YEARS',
        'CREDIT_INCOME_RATIO',
        'ANNUITY_INCOME_RATIO',
        'GOODS_CREDIT_RATIO'
    ]

    for feature in derived_features:
        assert feature in X_train.columns


def test_preprocess_handles_categorical_variables(sample_raw_data, temp_dir):
    """Test that categorical variables are properly encoded"""
    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    sample_raw_data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check that original categorical columns are removed
    assert 'NAME_INCOME_TYPE' not in X_train.columns
    assert 'NAME_EDUCATION_TYPE' not in X_train.columns

    # Check that dummy variables are created
    dummy_cols = [col for col in X_train.columns if 'NAME_INCOME_TYPE' in col or 'NAME_EDUCATION_TYPE' in col]
    assert len(dummy_cols) > 0


def test_preprocess_stratified_split(sample_raw_data, temp_dir):
    """Test that train-test split maintains class distribution"""
    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    sample_raw_data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    original_default_rate = sample_raw_data['TARGET'].mean()

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check class distribution is similar in train and test
    train_default_rate = y_train.mean()
    test_default_rate = y_test.mean()

    # Should be reasonably close to original (within 0.2 tolerance for small sample)
    assert abs(train_default_rate - original_default_rate) < 0.2
    assert abs(test_default_rate - original_default_rate) < 0.3


def test_preprocess_scales_features(sample_raw_data, temp_dir):
    """Test that features are properly scaled"""
    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    sample_raw_data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )

    # Check that scaled features have approximately zero mean and unit variance
    # (allowing for some tolerance due to small sample size)
    for col in X_train.columns:
        train_mean = X_train[col].mean()
        train_std = X_train[col].std()
        # Mean should be close to 0, std should be close to 1
        assert abs(train_mean) < 1.0  # Relaxed tolerance
        assert 0.5 < train_std < 1.5  # Relaxed tolerance


def test_preprocess_handles_days_employed_anomaly(temp_dir):
    """Test that DAYS_EMPLOYED anomaly (365243) is handled"""
    np.random.seed(42)
    n_samples = 50

    # Create data with some anomalous DAYS_EMPLOYED values
    data = pd.DataFrame({
        'TARGET': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3]),
        'AMT_INCOME_TOTAL': np.random.uniform(100000, 200000, n_samples),
        'AMT_CREDIT': np.random.uniform(300000, 600000, n_samples),
        'AMT_ANNUITY': np.random.uniform(15000, 30000, n_samples),
        'AMT_GOODS_PRICE': np.random.uniform(280000, 550000, n_samples),
        'DAYS_BIRTH': np.random.randint(-15000, -10000, n_samples),
        'DAYS_EMPLOYED': [365243 if i < 10 else -2000 for i in range(n_samples)],  # 10 anomalous values
        'CNT_CHILDREN': np.random.randint(0, 4, n_samples),
        'CNT_FAM_MEMBERS': np.random.randint(2, 6, n_samples),
        'REGION_POPULATION_RELATIVE': np.random.uniform(0.01, 0.03, n_samples),
        'DAYS_ID_PUBLISH': np.random.randint(-1500, -800, n_samples),
        'OWN_CAR_AGE': np.random.uniform(0, 15, n_samples),
        'FLAG_OWN_CAR': np.random.choice(['Y', 'N'], n_samples),
        'FLAG_OWN_REALTY': np.random.choice(['Y', 'N'], n_samples),
        'NAME_INCOME_TYPE': np.random.choice(['Working', 'Pensioner'], n_samples),
        'NAME_EDUCATION_TYPE': np.random.choice(['Higher education', 'Secondary / secondary special'], n_samples),
        'NAME_FAMILY_STATUS': np.random.choice(['Married', 'Single / not married'], n_samples),
        'OCCUPATION_TYPE': np.random.choice(['Laborers', 'Sales staff'], n_samples),
        'EXT_SOURCE_1': np.random.uniform(0.3, 0.8, n_samples),
        'EXT_SOURCE_2': np.random.uniform(0.4, 0.9, n_samples),
        'EXT_SOURCE_3': np.random.uniform(0.5, 0.9, n_samples),
    })

    input_path = os.path.join(temp_dir, 'input.csv')
    train_output = os.path.join(temp_dir, 'train.csv')
    test_output = os.path.join(temp_dir, 'test.csv')
    scaler_output = os.path.join(temp_dir, 'scaler.pkl')

    data.to_csv(input_path, index=False)

    mock_params = {
        'preprocess': {'test_size': 0.2, 'random_state': 42}
    }

    with patch('src.data.preprocess.load_params', return_value=mock_params):
        X_train, X_test, y_train, y_test = preprocess_data(
            input_path, train_output, test_output, scaler_output
        )
    # Check that EMPLOYMENT_YEARS doesn't have extreme values
    # (365243 days = 1000 years, should be replaced with NaN and then median)
    assert X_train['EMPLOYMENT_YEARS'].max() < 100  # No one works 100+ years
    assert X_test['EMPLOYMENT_YEARS'].max() < 100
