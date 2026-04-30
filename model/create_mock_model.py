import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.mock_model import MockFraudModel

def create_mock_model(data_path):
    """Create a mock fraud detection model for demonstration"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}")
    print(f"Fraud distribution: {df['is_fraud'].value_counts()}")
    
    # Create and save mock model
    model = MockFraudModel()
    
    # Save model
    print("Saving mock model...")
    joblib.dump(model, 'model/mock_model.pkl')
    
    # Create feature importance for display
    feature_importance = pd.DataFrame({
        'feature': ['device_risk_score', 'ip_risk_score', 'amount', 'hour', 'transaction_type', 'merchant_category', 'country'],
        'importance': [0.30, 0.30, 0.20, 0.10, 0.05, 0.03, 0.02]
    })
    
    # Save feature importance
    joblib.dump(feature_importance, 'model/feature_importance.pkl')
    
    print("Mock model created successfully!")
    return model

if __name__ == "__main__":
    model = create_mock_model('data/synthetic_fraud_dataset.csv')
    print("\nMock model ready for demonstration!")
