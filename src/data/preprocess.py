"""
Preprocess Home Credit dataset for model training
"""
import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib


def load_params():
    """Load parameters from params.yaml"""
    with open('params.yaml', 'r') as f:
        params = yaml.safe_load(f)
    return params


def preprocess_data(input_path: str, train_output: str, test_output: str, scaler_output: str):
    """
    Preprocess Home Credit data
    """
    params = load_params()
    
    print("Loading data...")
    df = pd.read_csv(input_path)
    
    print("Preprocessing data...")
    
    # Select relevant features
    selected_features = [
        'TARGET',
        'AMT_INCOME_TOTAL',
        'AMT_CREDIT',
        'AMT_ANNUITY',
        'AMT_GOODS_PRICE',
        'DAYS_BIRTH',
        'DAYS_EMPLOYED',
        'CNT_CHILDREN',
        'CNT_FAM_MEMBERS',
        'REGION_POPULATION_RELATIVE',
        'DAYS_ID_PUBLISH',
        'OWN_CAR_AGE',
        'FLAG_OWN_CAR',
        'FLAG_OWN_REALTY',
        'NAME_INCOME_TYPE',
        'NAME_EDUCATION_TYPE',
        'NAME_FAMILY_STATUS',
        'OCCUPATION_TYPE',
        'EXT_SOURCE_1',
        'EXT_SOURCE_2',
        'EXT_SOURCE_3',
    ]
    
    available_features = [f for f in selected_features if f in df.columns]
    df = df[available_features].copy()
    
    print(f"Selected {len(available_features)} features")
    
    # Handle DAYS_EMPLOYED anomaly
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    # Create new features
    df['AGE_YEARS'] = abs(df['DAYS_BIRTH']) / 365
    df['EMPLOYMENT_YEARS'] = abs(df['DAYS_EMPLOYED']) / 365
    df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1)
    df['GOODS_CREDIT_RATIO'] = df['AMT_GOODS_PRICE'] / (df['AMT_CREDIT'] + 1)
    
    # Drop original DAYS columns
    df = df.drop(['DAYS_BIRTH', 'DAYS_EMPLOYED'], axis=1, errors='ignore')
    
    # Handle categorical variables
    categorical_cols = df.select_dtypes(include=['object']).columns
    
    for col in categorical_cols:
        if col != 'TARGET':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(col, axis=1)
    
    # Handle missing values
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    numerical_cols = [col for col in numerical_cols if col != 'TARGET']
    
    for col in numerical_cols:
        df[col] = df[col].fillna(df[col].median())
    
    # Remove any remaining rows with missing target
    df = df.dropna(subset=['TARGET'])
    
    print(f"Final shape after preprocessing: {df.shape}")
    
    # Separate features and target
    X = df.drop('TARGET', axis=1)
    y = df['TARGET']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=params['preprocess']['test_size'],
        random_state=params['preprocess']['random_state'],
        stratify=y
    )
    
    print(f"Training set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    print(f"Training default rate: {y_train.mean():.2%}")
    print(f"Test default rate: {y_test.mean():.2%}")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    # Save processed data
    train_data = pd.concat([X_train_scaled, y_train], axis=1)
    test_data = pd.concat([X_test_scaled, y_test], axis=1)
    
    Path(train_output).parent.mkdir(parents=True, exist_ok=True)
    train_data.to_csv(train_output, index=False)
    test_data.to_csv(test_output, index=False)
    
    # Save scaler
    Path(scaler_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, scaler_output)
    
    print(f"\nPreprocessed data saved:")
    print(f"  Training: {train_output}")
    print(f"  Test: {test_output}")
    print(f"  Scaler: {scaler_output}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test


if __name__ == "__main__":
    preprocess_data(
        input_path="data/raw/home_credit_raw.csv",
        train_output="data/processed/train.csv",
        test_output="data/processed/test.csv",
        scaler_output="models/scaler.pkl"
    )

