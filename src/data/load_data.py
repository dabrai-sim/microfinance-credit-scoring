
"""
Load Home Credit Default Risk dataset
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path


def load_home_credit_data(input_path: str, output_path: str):
    """
    Load Home Credit Default Risk dataset and prepare it for processing
    
    Args:
        input_path: Path to the raw application_train.csv file
        output_path: Path to save the loaded data
    """
    print("Loading Home Credit Default Risk dataset...")
    
    # Load the main application training data
    df = pd.read_csv(input_path)
    
    print(f"Dataset loaded successfully!")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print(f"\nTarget distribution:")
    print(df['TARGET'].value_counts(normalize=True))
    
    # Basic info
    print(f"\nMissing values (top 10):")
    missing = df.isnull().sum()
    print(missing[missing > 0].sort_values(ascending=False).head(10))
    
    # Save the raw data
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"\nData saved to: {output_path}")
    print(f"Total records: {len(df)}")
    print(f"Default rate: {df['TARGET'].mean():.2%}")


if __name__ == "__main__":
    # Default paths
    input_path = "data/raw/application_train.csv"
    output_path = "data/raw/home_credit_raw.csv"
    
    load_home_credit_data(input_path, output_path)
